// NFDRS4 (firelab/NFDRS4) KBDI harness, following ../PROTOCOL.md.
//
// Variants (all single-threaded; the library is built without OpenMP):
//   kernel      NFDRS4::iCalcKBDI driven over every day, carrying CummPrecip/YKBDI/KBDI
//               exactly as NFDRS4::UpdateDaily does around the call.
//   api         NFDRS4::UpdateDaily, one NFDRS4 object per cell, KBDI read from the
//               public member after each day.
//   api_hourly  NFDRS4::Update (hourly), 24 calls per day, one object per cell.
//
// Units are the library's own (inches, deg F, integer KBDI in hundredths of an inch);
// output values are KBDI * 0.254 (mm).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "nfdrs4.h"

using Clock = std::chrono::steady_clock;

static std::vector<double> read_f64(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(2); }
    size_t n = (size_t)f.tellg() / 8;
    std::vector<double> v(n);
    f.seekg(0);
    f.read((char*)v.data(), n * 8);
    return v;
}

static size_t meta_int(const std::string& meta, const char* key) {
    size_t at = meta.find(std::string("\"") + key + "\"");
    at = meta.find(':', at) + 1;
    return (size_t)std::strtoull(meta.c_str() + at, nullptr, 10);
}

struct Date { int year, month, day, julian; };

// consecutive calendar days from 1991-01-01 (the fixture's first day)
static std::vector<Date> make_dates(size_t days) {
    static const int dim[12] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    std::vector<Date> d(days);
    int y = 1991, m = 1, dd = 1, j = 1;
    for (size_t i = 0; i < days; i++) {
        d[i] = {y, m, dd, j};
        bool leap = (y % 4 == 0 && (y % 100 != 0 || y % 400 == 0));
        int len = dim[m - 1] + ((m == 2 && leap) ? 1 : 0);
        j++;
        if (++dd > len) { dd = 1; if (++m > 12) { m = 1; y++; j = 1; } }
    }
    return d;
}

static const double kLat = 45.0;
static const char kFuelModel = 'Y';
static const int kSlopeClass = 1;
// constant plausible values for the inputs KBDI does not use
static const double kRH = 40.0, kMinRH = 20.0, kWS = 10.0;
static const double kMC1 = 8.0, kMC10 = 9.0, kMC100 = 12.0, kMC1000 = 15.0, kFuelTempC = 20.0;

static void run_cells(NFDRS4& nf, const std::vector<double>& pin, const std::vector<double>& tf,
                      const std::vector<double>& map_in, size_t c0, size_t c1, size_t days,
                      std::vector<double>& out) {
    for (size_t c = c0; c < c1; c++) {
        nf.SetStartKBDI(0);   // KBDI = YKBDI = StartKBDI = 0
        nf.CummPrecip = 0.0;  // empty wet spell (Init does the same)
        const double* p = &pin[c * days];
        const double* t = &tf[c * days];
        double* o = &out[c * days];
        const double map = map_in[c];
        for (size_t d = 0; d < days; d++) {
            // same statements as NFDRS4::UpdateDaily
            nf.KBDI = nf.iCalcKBDI(p[d], (int)t[d], nf.CummPrecip, nf.YKBDI, map);
            nf.YKBDI = nf.KBDI;
            o[d] = nf.KBDI * 0.254;
        }
    }
}

static std::vector<double> run_kernel(NFDRS4& nf, const std::vector<double>& pin,
                                      const std::vector<double>& tf, const std::vector<double>& map_in,
                                      size_t cells, size_t days) {
    std::vector<double> out(cells * days);
    run_cells(nf, pin, tf, map_in, 0, cells, days, out);
    return out;
}

// harness-level parallelism (not an NFDRS4 feature): contiguous cell blocks, one
// NFDRS4 object per thread (iCalcKBDI keeps no shared state), threads spawned per call
static std::vector<double> run_kernel_mt(std::vector<NFDRS4>& nfs, const std::vector<double>& pin,
                                         const std::vector<double>& tf, const std::vector<double>& map_in,
                                         size_t cells, size_t days) {
    std::vector<double> out(cells * days);
    size_t n = nfs.size();
    std::vector<std::thread> pool;
    for (size_t k = 0; k < n; k++)
        pool.emplace_back([&, k] { run_cells(nfs[k], pin, tf, map_in, k * cells / n, (k + 1) * cells / n, days, out); });
    for (auto& th : pool) th.join();
    return out;
}

static std::vector<double> run_api(const std::vector<double>& pin, const std::vector<double>& tf,
                                   const std::vector<double>& map_in, const std::vector<Date>& dates,
                                   size_t cells, size_t days) {
    std::vector<double> out(cells * days);
    for (size_t c = 0; c < cells; c++) {
        NFDRS4 nf(kLat, kFuelModel, kSlopeClass, map_in[c], true, true, false);
        nf.SetStartKBDI(0);
        const double* p = &pin[c * days];
        const double* t = &tf[c * days];
        double* o = &out[c * days];
        for (size_t d = 0; d < days; d++) {
            const Date& dt = dates[d];
            nf.UpdateDaily(dt.year, dt.month, dt.day, dt.julian, t[d] - 15.0, t[d] - 30.0, t[d], kRH, kMinRH,
                           p[d], kWS, kMC1, kMC10, kMC100, kMC1000, kFuelTempC, false);
            o[d] = nf.KBDI * 0.254;
        }
    }
    return out;
}

// Hourly path. The regular observation hour (13) triggers the daily KBDI step with the
// trailing-24-hour max temperature and precipitation sum. Hours 14 (previous day) .. 13
// are in that window, so each day's precipitation goes in hour 13 and the day's Tmax in
// hour 13. Hours before 13 are 20 F cooler than that day's Tmax; hours after 13 belong to the
// next day's window, so they are 20 F cooler than the next day's Tmax. The window max/sum then
// equals the day's values exactly.
static std::vector<double> run_api_hourly(const std::vector<double>& pin, const std::vector<double>& tf,
                                          const std::vector<double>& map_in, const std::vector<Date>& dates,
                                          size_t cells, size_t days) {
    std::vector<double> out(cells * days);
    for (size_t c = 0; c < cells; c++) {
        NFDRS4 nf(kLat, kFuelModel, kSlopeClass, map_in[c], true, true, false);
        nf.SetStartKBDI(0);
        const double* p = &pin[c * days];
        const double* t = &tf[c * days];
        double* o = &out[c * days];
        for (size_t d = 0; d < days; d++) {
            const Date& dt = dates[d];
            for (int h = 0; h < 24; h++) {
                double temp = (h == 13) ? t[d] : (h < 13 ? t[d] : t[std::min(d + 1, days - 1)]) - 20.0;
                double pcp = (h == 13) ? p[d] : 0.0;
                double sr = (h >= 8 && h <= 17) ? 500.0 : 0.0;
                nf.Update(dt.year, dt.month, dt.day, h, temp, kRH, pcp, sr, kWS, false);
            }
            o[d] = nf.KBDI * 0.254;
        }
    }
    return out;
}

int main(int argc, char** argv) {
    std::string workload, variant = "kernel", timings = "out/timings.jsonl", values;
    size_t threads = 1;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string f = argv[i], v = argv[i + 1];
        if (f == "--workload") workload = v;
        else if (f == "--variant") variant = v;
        else if (f == "--timings") timings = v;
        else if (f == "--values") values = v;
        else if (f == "--threads") threads = (size_t)std::strtoull(v.c_str(), nullptr, 10);
        else { std::fprintf(stderr, "unknown flag %s\n", f.c_str()); return 2; }
    }
    if (workload.empty()) { std::fprintf(stderr, "--workload is required\n"); return 2; }
    std::string meta;
    { std::ifstream f(workload + "/meta.json"); std::stringstream ss; ss << f.rdbuf(); meta = ss.str(); }
    size_t cells = meta_int(meta, "cells"), days = meta_int(meta, "days");
    std::vector<double> pin = read_f64(workload + "/precip_in.bin");
    std::vector<double> tf = read_f64(workload + "/tmax_f.bin");
    std::vector<double> map_in = read_f64(workload + "/map_in.bin");
    std::vector<Date> dates = make_dates(days);
    std::string wname = workload;
    while (!wname.empty() && (wname.back() == '/' || wname.back() == '\\')) wname.pop_back();
    wname = wname.substr(wname.find_last_of("/\\") + 1);

    NFDRS4 kernel_obj;  // used by the kernel variant only
    if (variant != "kernel_mt") threads = 1;
    threads = std::max<size_t>(1, std::min(threads, cells));
    std::vector<NFDRS4> mt_objs(variant == "kernel_mt" ? threads : 0);
    std::string notes;
    auto call = [&]() -> std::vector<double> {
        if (variant == "kernel") return run_kernel(kernel_obj, pin, tf, map_in, cells, days);
        if (variant == "kernel_mt") return run_kernel_mt(mt_objs, pin, tf, map_in, cells, days);
        if (variant == "api") return run_api(pin, tf, map_in, dates, cells, days);
        if (variant == "api_hourly") return run_api_hourly(pin, tf, map_in, dates, cells, days);
        std::fprintf(stderr, "unknown variant %s\n", variant.c_str());
        std::exit(2);
    };
    if (variant == "kernel") notes = "NFDRS4::iCalcKBDI (public) driven per day, state in public members; start KBDI 0 via SetStartKBDI(0); int KBDI hundredths of inch";
    else if (variant == "kernel_mt") notes = "kernel on contiguous cell blocks, one NFDRS4 object and one std::thread per block, spawned per call (harness-level parallelism)";
    else if (variant == "api") notes = "NFDRS4::UpdateDaily per day, one object per cell built inside the timed call; extra: live fuel moisture, load transfer, SC/ERC/BI/IC";
    else notes = "NFDRS4::Update hourly, 24 calls/day, one object per cell built inside the timed call; extra: Nelson dead fuel moisture, live FM, indices every hour";
    auto checksum = [](const std::vector<double>& v) { double s = 0; for (double x : v) s += x; return s; };

    auto t0 = Clock::now();
    std::vector<double> last = call();
    long long first_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - t0).count();
    auto warm = Clock::now();
    int wc = 1;
    while (wc < 5 || std::chrono::duration<double>(Clock::now() - warm).count() < 0.5) { last = call(); wc++; }

    size_t min_samples = (cells == 1) ? 200 : 15;
    std::vector<long long> samples;
    double sum = 0;
    auto started = Clock::now();
    for (;;) {
        auto a = Clock::now();
        last = call();
        auto b = Clock::now();
        samples.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count());
        sum = checksum(last);
        double el = std::chrono::duration<double>(Clock::now() - started).count();
        if ((samples.size() >= min_samples && el >= 3.0) || (el >= 60.0 && samples.size() >= 3)) break;
    }

    if (!values.empty()) {
        size_t sl = values.find_last_of("/\\");
        if (sl != std::string::npos) {
            std::string cmd = "mkdir \"" + values.substr(0, sl) + "\" 2>nul";
            for (char& ch : cmd) if (ch == '/') ch = '\\';
            std::system(cmd.c_str());
        }
        std::ofstream f(values, std::ios::binary);
        f.write((const char*)last.data(), last.size() * 8);
    }
    std::ostringstream line;
    line << "{\"impl\": \"nfdrs4\", \"variant\": \"" << variant << "\", \"workload\": \"" << wname
         << "\", \"cells\": " << cells << ", \"days\": " << days << ", \"threads\": " << threads << ", \"samples_ns\": [";
    for (size_t i = 0; i < samples.size(); i++) line << (i ? "," : "") << samples[i];
    line.precision(17);
    line << "], \"first_call_ns\": " << first_ns << ", \"checksum\": " << sum << ", \"notes\": \"" << notes << "\"}\n";
    { std::ofstream f(timings, std::ios::binary | std::ios::app); f << line.str(); }
    std::vector<long long> s = samples; std::sort(s.begin(), s.end());
    std::fprintf(stderr, "%s %s: median %.4f ms, min %.4f ms, n=%zu, checksum %.6f\n", variant.c_str(), wname.c_str(),
                 s[s.size() / 2] / 1e6, s[0] / 1e6, samples.size(), sum);
    return 0;
}
