#ifndef __ICP_FOLLOWING__
#define __ICP_FOLLOWING__

#include <boost/shared_ptr.hpp>

#include "tipos_basicos.h"

ICPResult follow (boost::python::object source_cloud,
                  boost::python::object target_cloud);

void export_follow();

#endif //__ICP_FOLLOWING__
