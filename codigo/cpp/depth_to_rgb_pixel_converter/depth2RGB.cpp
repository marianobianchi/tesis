
struct CharRGB {
	unsigned char R, G, B;
        CharRGB() : R(0),G(0),B(0){}; // Se necesita para que compile con boost::python
	CharRGB(unsigned char _R,unsigned char _G,unsigned char _B) : R(_R),G(_G),B(_B) {}
};


CharRGB hsv2rgb(float H, float S, float V){

	unsigned char R, G, B;

	if ( S == 0 )                       //HSV from 0 to 1
	{
		R = V * 255;
		G = V * 255;
		B = V * 255;
	}
	else
	{
		float var_h = H * 6;
		if ( var_h == 6 ) var_h = 0;      //H must be < 1
		float var_i = int( var_h );             //Or ... var_i = floor( var_h )
		float var_1 = V * ( 1 - S );
		float var_2 = V * ( 1 - S * ( var_h - var_i ) );
		float var_3 = V * ( 1 - S * ( 1 - ( var_h - var_i ) ) );
		float var_r, var_g, var_b;

		if      ( var_i == 0 ) { var_r = V     ; var_g = var_3 ; var_b = var_1; }
		else if ( var_i == 1 ) { var_r = var_2 ; var_g = V     ; var_b = var_1; }
		else if ( var_i == 2 ) { var_r = var_1 ; var_g = V     ; var_b = var_3; }
		else if ( var_i == 3 ) { var_r = var_1 ; var_g = var_2 ; var_b = V;     }
		else if ( var_i == 4 ) { var_r = var_3 ; var_g = var_1 ; var_b = V;     }
		else                   { var_r = V     ; var_g = var_1 ; var_b = var_2; }

		R = var_r * 255;                  //RGB results from 0 to 255
		G = var_g * 255;
		B = var_b * 255;
	}

	return CharRGB(R, G, B);
}


CharRGB depth2RGB( unsigned short depth )
{

	// The Kinect depth sensor range is: minimum 800mm and maximum 4000mm.
	// The Kinect for Windows Hardware can however be switched to Near Mode which provides a range of 500mm to 3000mm instead of the Default range

	static const float MIN_RANGE = 700.0f;
	static const float MAX_RANGE = 4200.0f;

	if (depth == 0)
	{
		return CharRGB(0,0,0);
	}

	if (depth < MIN_RANGE)
	{
		return CharRGB(255,255,0);
	}

	if (depth > MAX_RANGE)
	{
		return CharRGB(255,255,255);
	}

	float H = (float)depth;
	H = (H-MIN_RANGE)/(MAX_RANGE-MIN_RANGE);

	//float S = (float)((depth >>4) & 0x03);
	//S = S / (2 << 2);
	//S = (S * 0.25f) + 0.5f; // saturation range [0.5 - 0.75]
	float S = 0.45f;

	float V = (float)((depth >> 8) & 0x07);
	V = V / (1 << 3);
	V = 1-V;
	V = (V * 0.5f) + 0.25f; // saturation range [0.25 - 0.75]

	//float V = (float)(depth & 0x0F);
	//float V = 0.9f;

	return hsv2rgb(H,S,V);
}


#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>



BOOST_PYTHON_MODULE(depth_to_rgb)
{
    using namespace boost::python;

    def("depth_to_rgb", depth2RGB);

    class_<CharRGB>("CharRGB")
        .def_readwrite("red", &CharRGB::R)
        .def_readwrite("green", &CharRGB::G)
        .def_readwrite("blue", &CharRGB::B);

}
