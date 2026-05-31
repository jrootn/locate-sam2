# Locate-SAM2 paper — build with Tectonic (XeTeX engine)
MAIN = main
BUILD = build

# Override if your Tectonic lives elsewhere:
#   make TECTONIC=/path/to/tectonic
TECTONIC ?= /home/jroot/projects directory/application_system/tools/tectonic

.PHONY: all clean figures check-tectonic

all: check-tectonic $(BUILD)/$(MAIN).pdf

check-tectonic:
	@test -x "$(TECTONIC)" || (echo "Set TECTONIC= to your tectonic binary (see README)" && exit 1)

$(BUILD)/$(MAIN).pdf: $(MAIN).tex references.bib figures/*
	mkdir -p $(BUILD)
	"$(TECTONIC)" --print --keep-logs --outdir $(BUILD) $(MAIN).tex
	"$(TECTONIC)" --outdir $(BUILD) $(MAIN).tex

clean:
	rm -rf $(BUILD)

figures:
	@test -f figures/win_ref5466/ours_overlay.png || (echo "Copy figures from outputs/full_val/figures/ (see README)" && exit 1)
