module kbdibench

go 1.22

require (
	alpineworks.io/firewx v1.2.0
	alpineworks.io/firewx/nfdrs v0.0.0
)

// setup.ps1 rewrites these paths for the local clone and checks the pinned commit.
replace alpineworks.io/firewx => C:/Users/work/AppData/Local/Temp/claude/C--Users-work-Documents-GitHub-climate-indices/69841be7-6991-45ad-aa4d-5ab559761677/scratchpad/src/firewx

replace alpineworks.io/firewx/nfdrs => C:/Users/work/AppData/Local/Temp/claude/C--Users-work-Documents-GitHub-climate-indices/69841be7-6991-45ad-aa4d-5ab559761677/scratchpad/src/firewx/nfdrs
