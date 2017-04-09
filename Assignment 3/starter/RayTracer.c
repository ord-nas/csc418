/*
  CSC418 - RayTracer code - Winter 2017 - Assignment 3&4

  Written Dec. 9 2010 - Jan 20, 2011 by F. J. Estrada
  Freely distributable for adacemic purposes only.

  Uses Tom F. El-Maraghi's code for computing inverse
  matrices. You will need to compile together with
  svdDynamic.c

  You need to understand the code provided in
  this file, the corresponding header file, and the
  utils.c and utils.h files. Do not worry about
  svdDynamic.c, we need it only to compute
  inverse matrices.

  You only need to modify or add code in sections
  clearly marked "TO DO"
*/

#include "utils.h"
#include "time.h"

// A couple of global structures and data: An object list, a light list, and the
// maximum recursion depth
struct object3D *object_list;
struct pointLS *light_list;
int MAX_DEPTH;

double min(double a, double b) {
  if (a < b) {
    return a;
  } else {
    return b;
  }
}

void insertAreaLS(struct object3D *plane, double r, double g, double b, double rows, double cols, struct pointLS **light_list) {
  // Simulate an area light source by representing it as a grid of point light
  // sources. Uses the given plane object as the surface to cover with
  // lights. Inserts all resulting point sources into the given light_list. Uses
  // the given number of rows and cols of point sources.

  // Divide r, g, b values by the number of point sources, so that the *overall*
  // intensity of the area light source is as expected.
  double numPLS = rows * cols;
  r /= numPLS;
  g /= numPLS;
  b /= numPLS;
  
  // Strategy: initially create the point sources on the canonical plane, and
  // then transform them to their proper spot by using the plane equation.
  struct point3D p;
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      // Canonical plane has points between -1 and 1 on the x and y axes, and 0
      // on the z axis.
      p.px = col / (cols - 1.0) * 2.0 - 1.0;
      p.py = row / (rows - 1.0) * 2.0 - 1.0;
      p.pz = 0;
      p.pw = 1;
      // Transform p using the plane transformation
      matVecMult(plane->T, &p);
      // Now create and insert a point light source
      struct pointLS *l = newPLS(&p, r, g, b);
      printf("Inserting light source with colour: %f %f %f\n",
	     l->col.R, l->col.G, l->col.B);
      insertPLS(l, light_list);
    }
  }
}

void buildScene(void) {
  // Sets up all objects in the scene. This involves creating each object,
  // defining the transformations needed to shape and position it as
  // desired, specifying the reflectance properties (albedos and colours)
  // and setting up textures where needed.
  // Light sources must be defined, positioned, and their colour defined.
  // All objects must be inserted in the object_list. All light sources
  // must be inserted in the light_list.
  //
  // To create hierarchical objects:
  //   Copy the transform matrix from the parent node to the child, and
  //   apply any required transformations afterwards.
  //
  // NOTE: After setting up the transformations for each object, don't
  //       forget to set up the inverse transform matrix!

  struct object3D *o;
  struct pointLS *l;
  struct point3D p;

  ///////////////////////////////////////
  // TO DO: For Assignment 3 you have to use
  //        the simple scene provided
  //        here, but for Assignment 4 you
  //        *MUST* define your own scene.
  //        Part of your mark will depend
  //        on how nice a scene you
  //        create. Use the simple scene
  //        provided as a sample of how to
  //        define and position objects.
  ///////////////////////////////////////

  // Simple scene for Assignment 3:
  // Insert a couple of objects. A plane and two spheres
  // with some transformations.

  // Let's add a plane
  // Note the parameters: ra, rd, rs, rg, R, G, B, alpha, r_index, and shinyness)
  o=newPlane(.05,.75,.05,.05,.55,.8,.75,1,1,2);  // Note the plane is highly-reflective (rs=rg=.75) so we
  loadTexture(o, "smarties.ppm");
  double r, g, b;
  texMap(o->texImg, 0, 0, &r, &g, &b);
  printf("0 0 -> %f %f %f\n", r, g, b);
  texMap(o->texImg, 0, 1, &r, &g, &b);
  printf("0 1 -> %f %f %f\n", r, g, b);
  texMap(o->texImg, 1, 0, &r, &g, &b);
  printf("1 0 -> %f %f %f\n", r, g, b);
  texMap(o->texImg, .999, .999, &r, &g, &b);
  printf("1 1 -> %f %f %f\n", r, g, b);
  // should see some reflections if all is done properly.
  // Colour is close to cyan, and currently the plane is
  // completely opaque (alpha=1). The refraction index is
  // meaningless since alpha=1
  Scale(o,6,6,1);                                // Do a few transforms...
  RotateZ(o,PI/1.20);
  RotateX(o,PI/2.25);
  Translate(o,0,-3,10);
  invert(&o->T[0][0],&o->Tinv[0][0]);            // Very important! compute
  // and store the inverse
  // transform for this object!
  insertObject(o,&object_list);                  // Insert into object list

  // Let's add a couple spheres
  o=newSphere(.05,.95,.35,.35,1,.25,.25,1,1,30);
  //loadTexture(o, "smarties.ppm");
  Scale(o,.75,.5,1.5);
  RotateY(o,PI/2);
  Translate(o,-1.45,1.1,3.5);
  invert(&o->T[0][0],&o->Tinv[0][0]);
  insertObject(o,&object_list);

  o=newSphere(.05,.95,.95,.75,.75,.95,.55,1,1,30);
  //loadTexture(o, "smarties.ppm");
  Scale(o,.5,2.0,1.0);
  RotateZ(o,PI/1.5);
  Translate(o,1.75,1.25,5.0);
  invert(&o->T[0][0],&o->Tinv[0][0]);
  insertObject(o,&object_list);

  o=newCylinder(.05,.95,.95,.4,.95,.95,0,1,1,30);
  RotateZ(o,PI/1.5);
  RotateX(o,-PI/1.2);
  Translate(o,1.75,-1.5,10.0);
  invert(&o->T[0][0],&o->Tinv[0][0]);
  insertObject(o,&object_list);

  // Insert a single point light source.
  p.px=0;
  p.py=15.5;
  p.pz=-5.5;
  p.pw=1;
  l=newPLS(&p,.95,.95,.95);
  insertPLS(l,&light_list);
  
  /* // Insert an area light source. */
  /* o=newPlane(0, 0, 0, 0, 0, 0, 0, 0, 0, 0);  // Note this plane is just for */
  /* 					     // defining the area light source, */
  /* 					     // so its material properties don't */
  /* 					     // matter. */
  /* Scale(o,2,2,1); */
  /* RotateZ(o,PI/1.20); */
  /* RotateX(o,PI/2.25); */
  /* Translate(o,0,15.5,-5.5); */
  /* insertAreaLS(o, .95, .95, .95, 20, 20, &light_list); */
  /* free(o); */

  // End of simple scene for Assignment 3
  // Keep in mind that you can define new types of objects such as cylinders and parametric surfaces,
  // or, you can create code to handle arbitrary triangles and then define objects as surface meshes.
  //
  // Remember: A lot of the quality of your scene will depend on how much care you have put into defining
  //           the relflectance properties of your objects, and the number and type of light sources
  //           in the scene.
}

bool inShadow(struct object3D *obj, struct point3D* p, struct pointLS* light) {
  // This function returns true iff the given object obj, at point p, is in
  // shadow with respect to the given light source.
  
  // Construct a ray pointing from p to the light source.
  struct point3D delta = light->p0;
  subVectors(p, &delta);
  delta.pw = 0;
  struct ray3D* ray = newRay(p, &delta);
  
  // Now fire it off and see what it hits!
  double lambda, unused_a, unused_b;
  struct object3D *unused_object;
  struct point3D unused_p, unused_n;
  findFirstHit(ray, &lambda, obj, &unused_object, &unused_p, &unused_n, &unused_a, &unused_b);
  free(ray);
  // lambda < 0 means ray didn't interset anything (no shadow)
  // lambda > 1 means intersection happened *after* the light source (no shadow)
  // else, we have shadow
  return (lambda > 0 && lambda < 1);
}

void rtShade(struct object3D *obj, struct point3D *p, struct point3D *n, struct ray3D *ray, int depth, double a_tex, double b_tex, struct colourRGB *col) {
  // This function implements the shading model as described in lecture. It takes
  // - A pointer to the first object intersected by the ray (to get the colour properties)
  // - The coordinates of the intersection point (in world coordinates)
  // - The normal at the point
  // - The ray (needed to determine the reflection direction to use for the global component, as well as for
  //   the Phong specular component)
  // - The current recursion depth
  // - The (a,b) texture coordinates (meaningless unless texture is enabled)
  //
  // Returns:
  // - The colour for this ray (using the col pointer)
  //

  struct colourRGB tmp_col;      // Accumulator for colour components
  double R,G,B;                  // Colour for the object in R G and B

  // This will hold the colour as we process all the components of
  // the Phong illumination model
  tmp_col.R=0;
  tmp_col.G=0;
  tmp_col.B=0;
  
  // Not textured, use object colour
  if (obj->texImg==NULL) {
    R=obj->col.R;
    G=obj->col.G;
    B=obj->col.B;
  } else {
    // Get object colour from the texture given the texture coordinates (a,b), and the texturing function
    // for the object. Note that we will use textures also for Photon Mapping.
    obj->textureMap(obj->texImg,a_tex,b_tex,&R,&G,&B);
  }

  //////////////////////////////////////////////////////////////
  // TO DO: Implement this function. Refer to the notes for
  // details about the shading model.
  //////////////////////////////////////////////////////////////

  // Ambient Lighting
  tmp_col.R += obj->alb.ra * R;
  tmp_col.G += obj->alb.ra * G;
  tmp_col.B += obj->alb.ra * B;

  // Vector pointing from p to camera
  struct point3D b = ray->p0;
  subVectors(p, &b);
  normalize(&b);
  
  // For diffuse and specular lighting, we need to iterate over all light
  // sources
  struct pointLS* light = light_list;
  while (light != NULL) {
    // First get a unit vector pointing in direction of light source
    struct point3D s = light->p0;
    subVectors(p, &s);
    normalize(&s);
    // If we're lighting both front and back, and if n points in the opposite
    // direction to s, just flip n.
    struct point3D current_n = *n;
    if (obj->frontAndBack && dot(&current_n, &s) < 0) {
      current_n.px *= -1;
      current_n.py *= -1;
      current_n.pz *= -1;
    }
    double n_dot_s = dot(&current_n, &s);
    
    // Only do diffuse and specular lighting if this point on the object is
    // actually illuminated by the current light source.
    if (n_dot_s > 0 && !inShadow(obj, p, light)) {
      // Diffuse Lighting
      double diffuse_coefficient = obj->alb.rd * n_dot_s;
      tmp_col.R += diffuse_coefficient * R * light->col.R;
      tmp_col.G += diffuse_coefficient * G * light->col.G;
      tmp_col.B += diffuse_coefficient * B * light->col.B;
    
      // Specular lighting
      struct point3D r;
      r.px = -s.px + 2*n_dot_s*current_n.px;
      r.py = -s.py + 2*n_dot_s*current_n.py;
      r.pz = -s.pz + 2*n_dot_s*current_n.pz;
      r.pw = 0.0;
      double specular_coefficient = obj->alb.rs * max(0, pow(dot(&r, &b), obj->shinyness));
      tmp_col.R += specular_coefficient * light->col.R;
      tmp_col.G += specular_coefficient * light->col.G;
      tmp_col.B += specular_coefficient * light->col.B;
    }
    
    light = light->next;
  }

  // Now launch a reflected ray
  // If we're lighting both front and back, and if n points in the opposite
  // direction to b, just flip n.
  struct point3D current_n = *n;
  if (obj->frontAndBack && dot(&current_n, &b) < 0) {
    current_n.px *= -1;
    current_n.py *= -1;
    current_n.pz *= -1;
  }
  double n_dot_b = dot(&current_n, &b);
  if (n_dot_b > 0) {
    struct point3D r;
    r.px = -b.px + 2*n_dot_b*current_n.px;
    r.py = -b.py + 2*n_dot_b*current_n.py;
    r.pz = -b.pz + 2*n_dot_b*current_n.pz;
    r.pw = 0.0;
    struct ray3D* reflected_ray = newRay(p, &r);
    struct colourRGB reflected_col;
    rayTrace(reflected_ray, depth+1, &reflected_col, obj);
    if (reflected_col.R >= 0 && reflected_col.G >= 0 && reflected_col.B >= 0) {
      tmp_col.R += obj->alb.rg * reflected_col.R;
      tmp_col.G += obj->alb.rg * reflected_col.G;
      tmp_col.B += obj->alb.rg * reflected_col.B;
    }
    free(reflected_ray);
  }    
  

  /* col->R = 0;//max(0, n->px); */
  /* col->G = 0;//max(0, n->py); */
  /* col->B = 0;//max(0, n->pz); */
  *col = tmp_col;  
  return;

}

void findFirstHit(struct ray3D *ray, double *lambda, struct object3D *Os, struct object3D **obj, struct point3D *p, struct point3D *n, double *a, double *b) {
  // Find the closest intersection between the ray and any objects in the scene.
  // It returns:
  //   - The lambda at the intersection (or < 0 if no intersection)
  //   - The pointer to the object at the intersection (so we can evaluate the colour in the shading function)
  //   - The location of the intersection point (in p)
  //   - The normal at the intersection point (in n)
  //
  // Os is the 'source' object for the ray we are processing, can be NULL, and is used to ensure we don't
  // return a self-intersection due to numerical errors for recursive raytrace calls.
  //

  /////////////////////////////////////////////////////////////
  // TO DO: Implement this function. See the notes for
  // reference of what to do in here
  /////////////////////////////////////////////////////////////
  // Now intersect it with all the objects
  struct object3D* current = object_list;
  *lambda = -1;
  *obj = NULL;
  while (current != NULL) {
    if (current == Os) {
      // Ignore the source object!
      current = current->next;
      continue;
    }
    // Transform the ray to canonical coordinates.
    struct ray3D t_ray = *ray;
    rayTransform(ray, &t_ray, current);
    // Get the intersection with the canonical object.
    double current_lambda;
    struct point3D current_p;
    struct point3D current_n;
    double current_a;
    double current_b;
    current->intersect(current, &t_ray, &current_lambda, &current_p,
		       &current_n, &current_a, &current_b);
    if (current_lambda > 0) {
      if (*lambda < 0 || current_lambda < *lambda) {
	// Yay, we're the closest object so far!
	*lambda = current_lambda;
	*obj = current;
	*p = current_p;
	*n = current_n;
	*a = current_a;
	*b = current_b;
      }
    }
    current = current->next;
  }

  if (*obj) {
    // Do some work to transform the point p and normal n.
    matVecMult((*obj)->T, p);
    normalTransform(n, n, *obj);
  }
}

void rayTrace(struct ray3D *ray, int depth, struct colourRGB *col, struct object3D *Os) {
  // Ray-Tracing function. It finds the closest intersection between
  // the ray and any scene objects, calls the shading function to
  // determine the colour at this intersection, and returns the
  // colour.
  //
  // Os is needed for recursive calls to ensure that findFirstHit will
  // not simply return a self-intersection due to numerical
  // errors. For the top level call, Os should be NULL. And thereafter
  // it will correspond to the object from which the recursive
  // ray originates.
  //

  double lambda;         // Lambda at intersection
  double a,b;            // Texture coordinates
  struct object3D *obj;  // Pointer to object at intersection
  struct point3D p;      // Intersection point
  struct point3D n;      // Normal at intersection
  struct colourRGB I;    // Colour returned by shading function

  // Max recursion depth reached. Return invalid colour.
  if (depth>MAX_DEPTH) {
    col->R=-1;
    col->G=-1;
    col->B=-1;
    return;
  }

  ///////////////////////////////////////////////////////
  // TO DO: Complete this function. Refer to the notes
  // if you are unsure what to do here.
  ///////////////////////////////////////////////////////

  findFirstHit(ray, &lambda, Os, &obj, &p, &n, &a, &b);
  if (lambda > 0) {
    rtShade(obj, &p, &n, ray, depth, a, b, col);
    //*col = obj->col;
  } else {
    col->R = -1;
    col->G = -1;
    col->B = -1;
  }
}

// Inverts the given 3x3 matrix in place. Returns true on success, false
// otherwise.
bool invert3x3Mat(float mat[3][3]) {
  float *U, *s, *V, *rv1;
  int singFlag, i;

  // Invert the affine transform
  U=NULL;
  s=NULL;
  V=NULL;
  rv1=NULL;
  singFlag=0;

  SVD(&mat[0][0],3,3,&U,&s,&V,&rv1);
  if (U==NULL||s==NULL||V==NULL) {
    // Can't invert!
    return false;
  }

  // Check for singular matrices...
  for (i=0;i<3;i++) if (*(s+i)<1e-9) singFlag=1;
  if (singFlag) {
    // Can't invert!
    return false;
  }

  // Compute and store inverse matrix
  InvertMatrix(U,s,V,3,&mat[0][0]);

  free(U);
  free(s);
  free(V);

  return true;
}

void launchRay(struct view* cam, double du, double dv, double i, double j, struct colourRGB* background, struct colourRGB* col) {
  // This function constructs a ray from the camera through the given pixel (i,
  // j) and returns the resulting colour. Pixel coordinates are floating point
  // to allow anti-aliasing to send rays that are more fine-grained than
  // pixel-level.
  double u = cam->wl + du * i;
  double v = cam->wt + dv * j;
  struct point3D d;
  d.px = u;
  d.py = v;
  d.pz = cam->f;
  d.pw = 0;

  // Transform d into world coordinates
  matVecMult(cam->C2W, &d);

  // Ray in world coordinates
  struct ray3D* ray = newRay(&(cam->e), &d);

  // Fire the ray and get the colour.
  int depth = 0;
  rayTrace(ray, depth, col, NULL);

  // If we got an invalid colour (because the ray didn't hit anything), just set
  // colour to the background colour.
  if (col->R < 0 || col->G < 0 || col->B < 0) {
    *col = *background;
  }
  
  free(ray);
}

int main(int argc, char *argv[]) {
  // Main function for the raytracer. Parses input parameters,
  // sets up the initial blank image, and calls the functions
  // that set up the scene and do the raytracing.
  struct image *im;      // Will hold the raytraced image
  struct view *cam;      // Camera and view for this scene
  int sx;                // Size of the raytraced image
  int antialiasing;      // Flag to determine whether antialiaing is enabled or disabled
  char output_name[1024];        // Name of the output file for the raytraced .ppm image
  struct point3D e;              // Camera view parameters 'e', 'g', and 'up'
  struct point3D g;
  struct point3D up;
  double du, dv;                 // Increase along u and v directions for pixel coordinates
  struct point3D pc,d;           // Point structures to keep the coordinates of a pixel and
  // the direction or a ray
  struct ray3D *ray;             // Structure to keep the ray from e to a pixel
  struct colourRGB col;          // Return colour for raytraced pixels
  struct colourRGB background;   // Background colour
  int i,j;                       // Counters for pixel coordinates
  unsigned char *rgbIm;
  int rgbArray[500][500][3];

  // Seed random number generator
  srand48(time(NULL));
  
  if (argc<5) {
    fprintf(stderr,"RayTracer: Can not parse input parameters\n");
    fprintf(stderr,"USAGE: RayTracer size rec_depth antialias output_name\n");
    fprintf(stderr,"   size = Image size (both along x and y)\n");
    fprintf(stderr,"   rec_depth = Recursion depth\n");
    fprintf(stderr,"   antialias = A single digit, 0 disables antialiasing. Anything else enables antialiasing\n");
    fprintf(stderr,"   output_name = Name of the output file, e.g. MyRender.ppm\n");
    exit(0);
  }
  sx=atoi(argv[1]);
  MAX_DEPTH=atoi(argv[2]);
  if (atoi(argv[3])==0) antialiasing=0; else antialiasing=1;
  strcpy(&output_name[0],argv[4]);

  fprintf(stderr,"Rendering image at %d x %d\n",sx,sx);
  fprintf(stderr,"Recursion depth = %d\n",MAX_DEPTH);
  if (!antialiasing) fprintf(stderr,"Antialising is off\n");
  else fprintf(stderr,"Antialising is on\n");
  fprintf(stderr,"Output file name: %s\n",output_name);

  object_list=NULL;
  light_list=NULL;

  // Allocate memory for the new image
  im=newImage(sx, sx);
  if (!im) {
    fprintf(stderr,"Unable to allocate memory for raytraced image\n");
    exit(0);
  }
  else rgbIm=(unsigned char *)im->rgbdata;

  ///////////////////////////////////////////////////
  // TO DO: You will need to implement several of the
  //        functions below. For Assignment 3, you can use
  //        the simple scene already provided. But
  //        for Assignment 4 you need to create your own
  //        *interesting* scene.
  ///////////////////////////////////////////////////
  buildScene();          // Create a scene. This defines all the
                         // objects in the world of the raytracer

  //////////////////////////////////////////
  // TO DO: For Assignment 3 you can use the setup
  //        already provided here. For Assignment 4
  //        you may want to move the camera
  //        and change the view parameters
  //        to suit your scene.
  //////////////////////////////////////////

  // Mind the homogeneous coordinate w of all vectors below. DO NOT
  // forget to set it to 1, or you'll get junk out of the
  // geometric transformations later on.

  // Camera center is at (0,0,-1)
  e.px=0;
  e.py=0;
  e.pz=-3;
  e.pw=1;

  // To define the gaze vector, we choose a point 'pc' in the scene that
  // the camera is looking at, and do the vector subtraction pc-e.
  // Here we set up the camera to be looking at the origin, so g=(0,0,0)-(0,0,-1)
  g.px=0;
  g.py=0;
  g.pz=1;
  g.pw=1;

  // Define the 'up' vector to be the Y axis
  up.px=0;
  up.py=1;
  up.pz=0;
  up.pw=1;

  // Set up view with given the above vectors, a 4x4 window,
  // and a focal length of -1 (why? where is the image plane?)
  // Note that the top-left corner of the window is at (-2, 2)
  // in camera coordinates.
  cam=setupView(&e, &g, &up, -3, -2, 2, 4);

  if (cam==NULL) {
    fprintf(stderr,"Unable to set up the view and camera parameters. Out of memory!\n");
    cleanup(object_list,light_list);
    deleteImage(im);
    exit(0);
  }

  // Set up background colour here
  background.R=0;
  background.G=0;
  background.B=0;

  // Do the raytracing
  //////////////////////////////////////////////////////
  // TO DO: You will need code here to do the raytracing
  //        for each pixel in the image. Refer to the
  //        lecture notes, in particular, to the
  //        raytracing pseudocode, for details on what
  //        to do here. Make sure you undersand the
  //        overall procedure of raytracing for a single
  //        pixel.
  //////////////////////////////////////////////////////
  du=cam->wsize/(sx);          // du and dv. In the notes in terms of wl and wr, wt and wb,
  dv=-cam->wsize/(sx);         // here we use wl, wt, and wsize. du=dv since the image is
  // and dv is negative since y increases downward in pixel
  // coordinates and upward in camera coordinates.

  fprintf(stderr,"View parameters:\n");
  fprintf(stderr,"Left=%f, Top=%f, Width=%f, f=%f\n",cam->wl,cam->wt,cam->wsize,cam->f);
  fprintf(stderr,"Camera to world conversion matrix (make sure it makes sense!):\n");
  printmatrix(cam->C2W);
  fprintf(stderr,"World to camera conversion matrix\n");
  printmatrix(cam->W2C);
  fprintf(stderr,"\n");

  fprintf(stderr,"Rendering row: ");
  // For each of the pixels in the image
  //#pragma omp parallel for
  for (j=0;j<sx;j++) {
    fprintf(stderr,"%d/%d, ",j,sx);
    for (i=0;i<sx;i++) {
      ///////////////////////////////////////////////////////////////////
      // TO DO - complete the code that should be in this loop to do the
      //         raytracing!
      ///////////////////////////////////////////////////////////////////

      struct colourRGB col;
      if (!antialiasing) {
	launchRay(cam, du, dv, i + 0.5, j + 0.5, &background, &col);
      } else {
	int numSteps = 4;
	// Divide this pixel into a grid of numSteps x numSteps, and fire a ray
	// into each grid cell. Then take the average colour.
	col.R = 0;
	col.G = 0;
	col.B = 0;
	for (int ii = 0; ii < numSteps; ++ii) {
	  for (int jj = 0; jj < numSteps; ++jj) {
	    struct colourRGB current_col;
	    // Choose a point randomly from inside the grid cell, to avoid moire
	    // patterns
	    double new_i = i + (ii + 0.5)/numSteps;
	    double new_j = j + (jj + 0.5)/numSteps;
	    launchRay(cam, du, dv, new_i, new_j, &background, &current_col);
	    //current_col.R = 1;
	    //current_col.G = 0;
	    //current_col.B = 0;
	    col.R += current_col.R;
	    col.G += current_col.G;
	    col.B += current_col.B;
	  }
	}
	col.R /= (numSteps*numSteps);
	col.G /= (numSteps*numSteps);
	col.B /= (numSteps*numSteps);
      }

      // Write the colour to the image
      rgbArray[j][i][0] = min(1.0, col.R) * 255 + 0.5;
      rgbArray[j][i][1] = min(1.0, col.G) * 255 + 0.5;
      rgbArray[j][i][2] = min(1.0, col.B) * 255 + 0.5;
      /* if (rgbArray[j][i][0] != 255) printf("\n\nERROR!\n\n"); */
      /* if (rgbArray[j][i][1] != 0) printf("\n\nERROR!\n\n"); */
      /* if (rgbArray[j][i][2] != 0) printf("\n\nERROR!\n\n"); */
      
    } // end for i
  } // end for j
  
  for (int i = 0; i < sx; ++i) {
    for (int j = 0; j < sx; ++j) {
      rgbIm[3*sx*j + 3*i + 0] = rgbArray[j][i][0];
      rgbIm[3*sx*j + 3*i + 1] = rgbArray[j][i][1];
      rgbIm[3*sx*j + 3*i + 2] = rgbArray[j][i][2];
    }
  }

  fprintf(stderr,"\nDone!\n");

  // Output rendered image
  imageOutput(im,output_name);

  // Exit section. Clean up and return.
  cleanup(object_list,light_list);               // Object and light lists
  deleteImage(im);                               // Rendered image
  free(cam);                                     // camera view
  exit(0);
}
