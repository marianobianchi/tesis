#include <iostream>
#include <assert.h>

#include "common.h"


using namespace std;


void test_flat_and_cloud_conversion(){
    pair<float, float> cloudXY;
    pair<int, int> flatXY;
    
    float depth = 1.3;
    
    for(int i=0;i<480;i++){
        for(int j=0;j<640;j++){
            cloudXY = from_flat_to_cloud(i, j, depth);
            flatXY = from_cloud_to_flat(cloudXY.first, cloudXY.second, depth);
            
            cout << " i = " << i << "    j = " << j << endl;
            cout << "ci = " << flatXY.first << "   cj = " << flatXY.second << endl;
            assert((flatXY.first -1) <= i <= (flatXY.first +1));
            assert((flatXY.second -1) <= j <= (flatXY.second +1));
        }
    }
}




int main(int argc, char** argv){

    test_flat_and_cloud_conversion();

}



