#ifndef __ICP_FOLLOWING__
#define __ICP_FOLLOWING__


ICPResult follow (IntPair top_left, IntPair bottom_right, std::string depth_fname, std::string source_cloud_fname, std::string target_cloud_fname);

void export_follow();

#endif //__ICP_FOLLOWING__
