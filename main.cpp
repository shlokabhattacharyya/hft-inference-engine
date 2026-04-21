#include <array>
#include <iostream>
#include <iomanip>

#include "engine.h"

int main() {
	std::cout << "hft_inference_engine v0.1.0 - C/C++ bridge test\n";

	std::array<float, 10> features = {
		0.001f,		// 1-tick return 
		0.003f,		// 5-tick return
		0.02f,		// spread
		0.0001f,	// normalized spread
		0.15f,		// book imbalance
		0.8f,		// signed trade flow
		1.2f,		// volume ratio
		0.005f,		// 10-tick momentum
		3.5f,		// tick intensity
		-0.001f		// VWAP deviation
	};

	std::array<float, 3> probs{};
	// run_inference(features.data, probs()); // uncomment once signature matches
	
	std::cout << "inference output (dummy for now): ";
	for (float p : probs){
		std::cout << std::fixed << std::setprecision(3) << p << " ";
	}
	std::cout << "\n";

	return 0;
}
