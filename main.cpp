/*
	Sequential C++ cnn for the mnist dataset credit to Can Boluk github.com/can1357/simple_cnn

	Updated by Chase Brown for use with Faces in the Wild images for runtime comparison with a 
	CUDA version CNN using the same CNN architecture.
	
	Convolution-->Activation-->Convolution-->Activation-->Flatten-->Activation

*/

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <time.h>
#include "byteswap.h"
#include "CNN/cnn.h"

using namespace std;

float train( vector<layer_t*>& layers, tensor_t<float>& data, tensor_t<float>& expected )
{
	for ( int i = 0; i < layers.size(); i++ )
	{
		if ( i == 0 )
			activate( layers[i], data );
		else
			activate( layers[i], layers[i - 1]->out );
	}

	tensor_t<float> grads = layers.back()->out - expected;

	for ( int i = layers.size() - 1; i >= 0; i-- )
	{
		if ( i == layers.size() - 1 )
			calc_grads( layers[i], grads );
		else
			calc_grads( layers[i], layers[i + 1]->grads_in );
	}

	for ( int i = 0; i < layers.size(); i++ )
	{
		fix_weights( layers[i] );
	}

	float err = 0;
	for ( int i = 0; i < grads.size.x * grads.size.y * grads.size.z; i++ )
	{
		float f = expected.data[i];
		if ( f > 0.5 )
			err += abs(grads.data[i]);
	}
	return err * 100;
}


void forward( vector<layer_t*>& layers, tensor_t<float>& data )
{
	for ( int i = 0; i < layers.size(); i++ )
	{
		if ( i == 0 )
			activate( layers[i], data );
		else
			activate( layers[i], layers[i - 1]->out );
	}
}

struct case_t
{
	tensor_t<float> data;
	tensor_t<float> out;
};

uint8_t* read_file( const char* szFile )
{
	ifstream file( szFile, ios::binary | ios::ate );
	streamsize size = file.tellg();
	file.seekg( 0, ios::beg );

	if ( size == -1 )
		return nullptr;

	uint8_t* buffer = new uint8_t[size];
	file.read( (char*)buffer, size );
	return buffer;
}

/* Updated to read in my Faces in the Wild binary training dataset */
vector<case_t> read_test_cases()
{
	vector<case_t> cases;

	//uint8_t* train_image = read_file( R"(C:\Users\chase\CLionProjects\seqtest\data\train-images-idx3-ubyte)" );
	//uint8_t* train_labels = read_file( R"(C:\Users\chase\CLionProjects\seqtest\data\train-labels-idx1-ubyte)" );

	uint8_t* train_image = read_file("data/train-images-idx3-ubyte");
	uint8_t* train_labels = read_file("data/train-labels-idx1-ubyte");

	uint32_t case_count = byteswap_uint32( *(uint32_t*)(train_image + 4) );

	for ( int i = 0; i < case_count; i++ )
	{
		case_t c {tensor_t<float>( 250, 250, 1 ), tensor_t<float>( 610, 1, 1 )};

		uint8_t* img = train_image + 16 + i * (250 * 250);
		uint8_t* label = train_labels + 8 + i;

		for ( int x = 0; x < 250; x++ )
			for ( int y = 0; y < 250; y++ )
				c.data( x, y, 0 ) = img[x + y * 250] / 255.f;

		for ( int b = 0; b < 610; b++ )
			c.out( b, 0, 0 ) = *label == b ? 1.0f : 0.0f;

		cases.push_back( c );
	}
	delete[] train_image;
	delete[] train_labels;

	return cases;
}

int main() {
    vector<case_t> cases = read_test_cases();

    vector<layer_t *> layers;

    conv_layer_t *layer1 = new conv_layer_t(1, 26, 9, cases[0].data.size);	// 250x250x1 (Kernel 26x26 stride 9) (9 Kernels)---> 225 x 225 x 9
    relu_layer_t *layer2 = new relu_layer_t(layer1->out.size); 				// Not actually relu. Changed to Sigmoid
    conv_layer_t *layer3 = new conv_layer_t(25, 25, 9, layer2->out.size);	// 225x225x9 (Kernel 25x25 stride 25) (9 Kernels) ---> 9 x 9 x 9
    relu_layer_t *layer4 = new relu_layer_t(layer3->out.size);				// Not actually relu. Changed to Sigmoid
    fc_layer_t *layer5 = new fc_layer_t(layer4->out.size, 610);				// 9 x 9 x 9 --> to output layer of 610 FC

    layers.push_back((layer_t *) layer1);
    layers.push_back((layer_t *) layer2);
    layers.push_back((layer_t *) layer3);
    layers.push_back((layer_t *) layer4);
    layers.push_back((layer_t *) layer5);


    float amse = 0;
    int ic = 0;

	clock_t start, end;
	double totaltime = 0.0;
	printf("Learning Phase\n");
    for (long ep = 0; ep < 5200;) {

        for (case_t &t : cases) {
			start = clock();
            float xerr = train(layers, t.data, t.out);
			end = clock();
            amse += xerr;


			totaltime += (end - start);
			printf("Epoch %ld Runtime: %.2f \n", ep+1 ,(totaltime/1e6));
            ep++;
            ic++;
        }
    }
    // end:
    return 0;
}