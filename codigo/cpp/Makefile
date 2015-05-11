DIRS = common alignment_prerejective depth_to_rgb_pixel_converter icp_follow
MAKEDIRS = $(DIRS:%=build-%)
CLEANDIRS = $(DIRS:%=clean-%)

all: $(MAKEDIRS)

$(DIRS): $(MAKEDIRS)
$(MAKEDIRS):
		$(MAKE) -C $(@:build-%=%)


clean: $(CLEANDIRS)
$(CLEANDIRS):
	$(MAKE) -C $(@:clean-%=%) clean
