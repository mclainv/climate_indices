# Normative original KBDI formulation

**Research date:** 2026-09-08
**Ticket:** [Establish the normative original KBDI formulation](https://github.com/monocongo/climate_indices/issues/787)
**Originating issue:** [Add Keetch–Byram Drought Index (KBDI)](https://github.com/monocongo/climate_indices/issues/785)
**Scope:** What Keetch and Byram (1968, SE-38) specify for the daily recurrence, and where later formulations diverge. This note does not choose the library's promised behavior; that is [#789](https://github.com/monocongo/climate_indices/issues/789). Production code is out of scope.

## Answer

The normative method is the **Crane-corrected English-unit Equation 18** of Keetch and Byram (1968), as printed in the November 1988 reprint of USDA Forest Service Research Paper SE-38, plus the paper's field bookkeeping rules for net rainfall, operation order, initialization, and bounds.

That is not the same as “whatever later fire-danger systems call KBDI.” Australian metric implementations (Crane 1982; Finkele et al. 2006; `xclim`) convert units, often round 8.00 inches to 200 mm instead of 203.2 mm, and replace 0.20 inch interception with 5 mm. Some operational US products add latitude, which SE-38 never uses. Computer code that copies the **1968-printed** Equation 18 without the 1988/Crane correction silently uses a different drying rate than the paper's own tables.

A `climate_indices` implementation that claims Keetch and Byram (1968) should:

1. Use English-unit Equation 18 with the **8.30** constant (1988 reprint / Crane / Alexander), not 0.830.
2. Apply net rainfall **before** the drought factor, using the consecutive-day 0.20-inch interception rule.
3. Treat mean annual rainfall as a climatological input in **inches**, continuously in Equation 18 (the five lookup tables were a field convenience).
4. Treat later SI/Australian/NFDRS variants as documented divergences, not as silent substitutions.

## Normative source identity

| Artifact | What it is |
| --- | --- |
| Keetch and Byram (1968), Res. Pap. SE-38 | Original publication. English units. Field tables plus Appendix Equation 18. |
| November 1988 reprint of SE-38 | Same paper; **Equations 15 and 18 are silently corrected**. The Treesearch PDF ([research.fs.usda.gov/treesearch/40](https://research.fs.usda.gov/treesearch/40)) is this reprint: cover line “November 1968 / Revised November 1988.” |
| Crane (1982), Alexander (1990, 1992) | Document that the **1968 typesetting** of Equation 18 used 0.830 (and Equation 15 used 2.113) instead of 8.30 and 0.2113. The drought-factor **tables were computed from the correct equation** and were never wrong. |

Alexander (1990) is explicit: the 1968 printed Equation 18 last numerator constant should be **8.30, not 0.830**; Equation 15 should be **0.2113, not 2.113**; a 1966 review draft already had the correct constants; Keetch and Byram indicated the 1988 reprinting corrected the misprints, with no errata notice. Using the uncorrected 1968 typesetting produces a drought factor that is always slightly too large and **accumulates** on rainless spells (Alexander 1990; Fujioka 1991).

**Normative computer equation (English units, 1988 reprint Eq. 18):**

\[
\mathrm{d}Q = \frac{(800 - Q)\,[0.968\,\exp(0.0486\,T) - 8.30]\,\mathrm{d}\tau}{1 + 10.88\,\exp(-0.0441\,R)}\times 10^{-3}
\]

where \(Q\) is yesterday's index after net-rain reduction (hundredths of an inch), \(T\) is today's maximum air temperature (°F), \(R\) is mean annual rainfall (inches), and \(\mathrm{d}\tau = 1\) day. The factor \(0.968\) multiplies only the exponential, then \(8.30\) is subtracted — not \(0.968\times(\exp(\cdot)-8.30)\).

A check against Table 4 (computed at \(R = 50\) in, \(T\) in 3 °F bands): for the 80–82 °F band at \(Q = 0\), Equation 18 at \(T = 81\) °F yields \(\mathrm{d}Q \approx 15.0\), matching the published table value of 15 after the paper's round-to-nearest-integer rule. The 0.830 typesetting does not.

## Explicit in Keetch and Byram (1968 / 1988)

Statements below are in the paper. Page references follow the 1988 reprint pagination used in the Treesearch PDF.

### Definition, scale, and units

- Drought index = net effect of evapotranspiration and precipitation on cumulative moisture deficiency in deep duff or upper soil layers.
- Field capacity of the soil–duff layer is **8.00 inches** of water available above wilting point (assumption 5; Appendix assumption 1). The index is that deficiency in **hundredths of an inch**, so the scale is **0 to 800**.
- Zero = no moisture deficiency (saturation). 800 = maximum drought. At any point, the index is the net rainfall (in hundredths of an inch) needed to return to saturation.
- Stages: 0–99 stage 0 (incipient), 100–199 stage 1, …, 700–800 stage 7. Local fire-control meaning of a stage is **not** universal; it must be determined locally.
- Mathematically 800 requires infinite time, so Equation 18 never reaches 800. Rounded table values **can** reach 800; once there, the increment is zero and the maximum cannot be exceeded.

### Inputs

- Daily **maximum air temperature**, or dry-bulb temperature at the time of the basic observation (°F). Record to the nearest degree; fractions of 0.5 round up.
- Total rainfall for the past **24 hours**, to the nearest 0.01 inch (include melted-snow water equivalent).
- **Long-term mean annual rainfall** of the rating area (inches). For field tables only, five bins: 10–19, 20–29, 30–39, 40–59, 60+. On a bin boundary (e.g. 19.50 or 39.50 in), use the **next higher** table. Equation 18 uses continuous \(R\).
- Latitude, humidity, wind, sunshine, and soil type are **not** inputs. The Appendix argues they are absorbed into the \(T\) and \(R\) empirical functions.

### Daily recurrence and operation order

Two steps, in this order:

1. **Reduce** yesterday's index by net rainfall, if any (one index point per 0.01 inch of net rain).
2. **Increase** that reduced value by today's drought factor from Equation 18 (or the matching table), using today's temperature and the reduced index.

Today's index = reduced yesterday's index + drought factor.

### Net rainfall / interception

- Subtract **0.20 inch** from the day's rain. If rain \(\le 0.20\) inch, net rain is 0.
- **Consecutive rainy days** with no canopy drying between showers: subtract 0.20 **only once**, on the day the **cumulative** rain of the wet spell first exceeds 0.20 inch. After that, all of each day's rain is net rain until the wet spell ends.
- Wet spell ends on the first 24-hour period with **no measurable rain**.
- **Snow:** while snow blankets the fuels, consider no drying, and transfer **all** measured water equivalent to net rainfall (no 0.20 subtraction while snow blankets).

Worked examples in the sample form (June 1966): isolated rains subtract 0.20 each; June 7–8 consecutive, 0.20 subtracted on the day the event total first exceeded 0.20; June 29–30 consecutive, 0.20 subtracted on day 1 (already > 0.20), all of day 2's rain is net.

### Initialization

- The index is cumulative. An observer **cannot automatically begin at zero**; zero may have occurred weeks, months, or a previous year earlier.
- Go back to a day when the upper soil layers were **reasonably certain to be saturated**, then compute forward day by day.
- Heavy-snow areas: normally safe to assume saturation **just after snowmelt**.
- Snow-free areas: go back to abundant rainfall such as **6 or 8 inches in a week**. The index must be very low, if not zero, at the end of that rainy period.

### Temperature behavior

- Drought-factor tables run **50–52 °F through 107+ °F**.
- Conclusion 1: drought development occurs only during extended little-or-no rainfall when daily maximum temperature is **50 °F or higher**.
- Conclusion 2: consistently high daily temperatures **averaging more than 70 °F** are required for appreciable drought starting from saturation.
- Appendix Equation 13 (the PET fit that feeds Equation 18) is stated to be valid from **50 °F to 110 °F**.

### Rainfall-event and bound behavior

- Reduction occurs only when 24-hour rainfall **exceeds 0.20 inch** (net rainfall). Isolated \(\le 0.20\) inch does not reduce the index, but a drought factor is still added that day.
- Index 800 cannot be exceeded. Drought increment at 800 is zero.

## Necessary interpretations (1968 is silent or only implicit)

These are required to implement a computer recurrence. They are **not** explicit numbered rules in SE-38. [#789](https://github.com/monocongo/climate_indices/issues/789) must record each one.

| Topic | Why an interpretation is required | Tightest reading of the paper |
| --- | --- | --- |
| Clip after rain | Net rain can exceed current \(Q\). The paper never says \(\max(0, Q - 100 P_\mathrm{net})\). Saturation is defined as 0. | Clip \(Q\) to \(\ge 0\) after the rain step. |
| Clip after drought factor | Equation 18 at \(Q < 800\) can overshoot 800 by a fraction of a point; tables were rounded. The paper says 800 cannot be exceeded. | Clip to \(\le 800\) after adding \(\mathrm{d}Q\). |
| \(T < 50\) °F | Equation 18's numerator \(0.968\exp(0.0486T)-8.30\) becomes **negative** near ~46 °F, which would wet the index without rain. Tables and conclusion 1 start drying at 50 °F. | Set \(\mathrm{d}Q = 0\) when \(T < 50\) °F (equivalently floor \(T\) at 50 °F before Equation 18). |
| \(T > 110\) °F | PET fit is only claimed to 110 °F; tables stop at 107+. | Use Equation 18 anyway, or cap \(T\) at 110 °F. Not specified. |
| Computer vs tables | Tables use binned \(R\), 3 °F bands, and integer drought factors. Equation 18 is continuous. Alexander (1990) notes equation vs table differences are expected for fire indices. | For a library, use **Equation 18**, not the five lookup tables. |
| Default initial state 0.0 | [#786](https://github.com/monocongo/climate_indices/issues/786) already requires an explicit initial state defaulting to 0.0. SE-38 forbids automatically starting at zero for a **scientific** record. | API may default to 0.0; documentation must state that this assumes saturation at the first timestep (Fujioka 1991 discusses start-up). |
| Missing values | SE-38 is a complete daily log. No missing-data rule. | Owned by the map: missing Tmax or precip makes that day and all following values missing. |
| “Measurable rain” | Ends a wet spell. US cooperative-observer measurable rain is typically 0.01 inch, which matches the recording precision. | Treat \(P > 0\) (or \(P \ge 0.01\) in) as measurable. |
| Time of observation | Max temperature **or** observation-time dry-bulb; 24-hour rain ending at the observation. No UTC/calendar-day rule. | Document that inputs are observation-day totals, not a prescribed hour. |
| Integer vs float index | Field form is integer hundredths. Equation 18 is real-valued. | Keep a floating recurrence; integer rounding is a display/table convenience. |

## Later formulations (do not adopt silently)

### 1. Uncorrected 1968 typesetting

Replace \(8.30\) with \(0.830\) in Equation 18. Daily error is small (Alexander 1990 Table 1: a few tenths of an index point per day) but **cumulative**. The 1988 reprint and the original tables are the correct original method. Do not implement the 1968 misprint.

### 2. Crane (1982) SI conversion — unit change, same physics

Alexander (1990) reprints Crane's SI form:

\[
\mathrm{d}Q = \frac{(203.2 - Q)\,[0.968\,\exp(0.0875\,T + 1.5552) - 8.30]\,\mathrm{d}\tau}{1 + 10.88\,\exp(-0.001736\,R)}\times 10^{-3}
\]

with \(Q\) in mm, \(T\) in °C, \(R\) in mm, net-rain threshold **0.20 in = 5.1 mm**, and 8.00 in = **203.2 mm**. The \(T\) and \(R\) coefficients are the Fahrenheit/inch conversions of Equation 18 (\(0.0486\times(1.8T_\mathrm{C}+32) = 0.08748T_\mathrm{C}+1.5552\); \(0.0441/25.4 \approx 0.001736\)).

This is a **unit conversion of the original**, not a new drying theory — provided 203.2 mm and 5.1 mm are kept.

### 3. Finkele et al. (2006) / Australian FFDI path — metric operational variant

Used by the Australian Bureau of Meteorology and by [`xclim.indices.fire.keetch_byram_drought_index`](https://xclim.readthedocs.io/en/stable/apidoc/xclim.indices.fire.html):

- SI drought-factor coefficients in the Crane family (\(0.0875T+1.5552\), \(0.00173\) or \(0.001736\) on \(R\)).
- Interception often **5.0 mm** (≈ 0.197 in), not 5.08/5.1 mm.
- Consecutive-day remaining-runoff of 5 mm, reset on a dry day (xclim's loop), which is the SE-38 wet-spell rule in millimetres.
- Cap at **200 mm** in Finkele et al. (2006) (8 inches approximated) versus **203.2 mm** in xclim (exact 8.00 in). xclim documents this as a deliberate departure from Finkele to match “the majority of the literature.”
- Default previous KBDI = 0 (xclim `kbdi0=None`).
- Intended as input to McArthur FFDI / Griffiths drought factor, not as a US fire-control log.

This is the most common **open-source** “KBDI.” It is **not** Keetch and Byram (1968) in native units, and the 200 vs 203.2 mm cap is a later choice.

### 4. Operational US products (Drought.gov, some NFDRS write-ups)

Drought.gov and several fact sheets list **station latitude** among KBDI inputs. SE-38 does not use latitude. Those products may also simplify net rain to “subtract 0.20 inch from each day independently,” dropping the consecutive-day exception. Treat as operational layers, not the 1968 method.

### 5. NFDRS relationship

SE-38 states the drought index is **not** a substitute for NFDRS spread or buildup moisture parameters; it is a slower, deeper moisture regime. Later inclusion of KBDI alongside NFDRS (Burgan 1988; Melton 1989) does not change the 1968 recurrence.

## Recommended contract seeds for #789

These are research recommendations, not decisions:

1. **Promise the 1988/Crane-corrected English Equation 18**, citing Keetch and Byram (1968) and Alexander (1990) for the typesetting history. Do not ship the 0.830 form.
2. **Keep native English units in the NumPy core** (index 0–800, \(T\) in °F, rain and \(R\) in inches) if the package is claiming the original method; an SI wrapper would be a documented conversion, not a second formulation.
3. **Net rain:** 0.20-inch interception with the consecutive-day wet-spell rule; clip \(Q\) to \([0, 800]\).
4. **Temperature floor:** \(\mathrm{d}Q = 0\) for \(T < 50\) °F.
5. **Initialization:** explicit `kbdi0` default 0.0 means “assume saturation at t0”; document the 6–8 inches-in-a-week / snowmelt start-up from the paper.
6. **Do not** take Finkele's 200 mm cap, xclim's 203.2 mm SI core, 5.0 mm interception, or Drought.gov latitude as the climate_indices method unless #789 explicitly chooses a named variant.

## What this ticket does not resolve

- Independent numerical oracle and fixture provenance ([#788](https://github.com/monocongo/climate_indices/issues/788)). The June 1966 sample form in SE-38 is a **worked bookkeeping example**, not a published machine-readable series.
- Exact public-API units, missing-data, and invalid-input errors ([#789](https://github.com/monocongo/climate_indices/issues/789)).
- NumPy / xarray / Dask file boundaries ([#790](https://github.com/monocongo/climate_indices/issues/790)).

## Sources

- Keetch, J. J.; Byram, G. M. (1968, rev. 1988). *A Drought Index for Forest Fire Control*. USDA Forest Service, Southeastern Forest Experiment Station, Research Paper SE-38. 35 p. <https://research.fs.usda.gov/treesearch/40>
- Crane, W. J. B. (1982). Computing grassland and forest fire behaviour, relative humidity and drought index by pocket calculator. *Australian Forestry* (see also Crane 1983, Institute of Foresters of Australia Newsletter 24(4): 27).
- Alexander, M. E. (1990). Computer calculation of the Keetch-Byram Drought Index—programmers beware! *Fire Management Notes* 51(4): 23–25. FRAMES: <https://www.frames.gov/catalog/10946>
- Alexander, M. E. (1992). The Keetch–Byram Drought Index: A Corrigendum. *Bulletin of the American Meteorological Society* 73(1): 61–62. <https://doi.org/10.1175/1520-0477-73.1.61>
- Finkele, K.; Mills, G. A.; Beard, G.; Jones, D. A. (2006). National gridded drought factors and comparison of two soil moisture deficit formulations used in prediction of Forest Fire Danger Index in Australia. *Australian Meteorological Magazine* 55: 183–197.
- Fujioka, F. M. (1991). Starting up the Keetch-Byram Drought Index. *Proceedings of the 11th Conference on Fire and Forest Meteorology*.
- xclim `keetch_byram_drought_index` documentation: <https://xclim.readthedocs.io/en/stable/apidoc/xclim.indices.fire.html> (Finkele path; 203.2 mm cap).
- Drought.gov KBDI product note (operational inputs including latitude): <https://www.drought.gov/data-maps-tools/keetch-byram-drought-index>
