all:
	g++ -std=c++11 main.cpp -o CNN_TEST
run:
	./CNN_TEST
clean:
	rm CNN_TEST
