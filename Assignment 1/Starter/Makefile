OBJS = Boids.o imgui_impl_glut.o imgui.o imgui_draw.o
CXXFLAGS = -O4 -g -fopenmp -L./lib -l3ds -I./include -lGL -lglut -lGLU -lm 


Boids: $(OBJS)
	g++ -o $@ $^ -L./lib -l3ds  -lGL -lglut -lGLU -lm

%.o: %.cpp
	g++ -c $(CXXFLAGS) -o $@ $<

clean:
	rm *.o Boids
