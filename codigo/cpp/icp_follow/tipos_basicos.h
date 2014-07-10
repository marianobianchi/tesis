#ifndef __TIPOS_BASICOS__
#define __TIPOS_BASICOS__

typedef std::pair<float,float> FloatPair;
typedef std::pair<int,int> IntPair;

typedef std::pair<FloatPair,FloatPair > DoubleFloatPair;
typedef std::pair<IntPair,IntPair > DoubleIntPair;

struct ICPResult {
    bool has_converged; // was found by icp?
    float score;        // icp score
    int size;           // size of the square frame containing the object
    int top;            // row
    int left;           // column
};


#endif //__TIPOS_BASICOS__
