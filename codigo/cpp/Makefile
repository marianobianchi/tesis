SUBDIRS = alignment_prerejective depth_to_rgb_pixel_converter icp_follow common

all: $(SUBDIRS)

$(SUBDIRS):
	$(MAKE) -C $@

.PHONY = $(SUBDIRS)
