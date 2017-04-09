/*
  utils.c - F.J. Estrada, Dec. 9, 2010

  Utilities for the ray tracer. You will need to complete
  some of the functions in this file. Look for the sections
  marked "TO DO". Be sure to read the rest of the file and
  understand how the entire code works.
*/

#include "utils.h"
#include <math.h>

// A useful 4x4 identity matrix which can be used at any point to
// initialize or reset object transformations
double eye4x4[4][4]={{1.0, 0.0, 0.0, 0.0},
		     {0.0, 1.0, 0.0, 0.0},
		     {0.0, 0.0, 1.0, 0.0},
		     {0.0, 0.0, 0.0, 1.0}};

/////////////////////////////////////////////
// Primitive data structure section
/////////////////////////////////////////////
struct point3D *newPoint(double px, double py, double pz) {
  // Allocate a new point structure, initialize it to
  // the specified coordinates, and return a pointer
  // to it.

  struct point3D *pt=(struct point3D *)calloc(1,sizeof(struct point3D));
  if (!pt) fprintf(stderr,"Out of memory allocating point structure!\n");
  else {
    pt->px=px;
    pt->py=py;
    pt->pz=pz;
    pt->pw=1.0;
  }
  return(pt);
}

struct pointLS *newPLS(struct point3D *p0, double r, double g, double b) {
  // Allocate a new point light sourse structure. Initialize the light
  // source to the specified RGB colour
  // Note that this is a point light source in that it is a single point
  // in space, if you also want a uniform direction for light over the
  // scene (a so-called directional light) you need to place the
  // light source really far away.

  struct pointLS *ls=(struct pointLS *)calloc(1,sizeof(struct pointLS));
  if (!ls) fprintf(stderr,"Out of memory allocating light source!\n");
  else {
    memcpy(&ls->p0,p0,sizeof(struct point3D));    // Copy light source location

    ls->col.R=r;                                  // Store light source colour and
    ls->col.G=g;                                  // intensity
    ls->col.B=b;
  }
  return(ls);
}

/////////////////////////////////////////////
// Ray and normal transforms
/////////////////////////////////////////////

/////////////////////////////////////////////
// Object management section
/////////////////////////////////////////////
struct object3D *newPlane(double ra, double rd, double rs, double rg, double r, double g, double b, double alpha, double r_index, double shiny, double rough) {
  // Intialize a new plane with the specified parameters:
  // ra, rd, rs, rg - Albedos for the components of the Phong model
  // r, g, b, - Colour for this plane
  // alpha - Transparency, must be set to 1 unless you are doing refraction
  // r_index - Refraction index if you are doing refraction.
  // shiny - Exponent for the specular component of the Phong model
  //
  // The plane is defined by the following vertices (CCW)
  // (1,1,0), (-1,1,0), (-1,-1,0), (1,-1,0)
  // With normal vector (0,0,1) (i.e. parallel to the XY plane)

  struct object3D *plane=(struct object3D *)calloc(1,sizeof(struct object3D));

  if (!plane) fprintf(stderr,"Unable to allocate new plane, out of memory!\n");
  else {
    plane->alb.ra=ra;
    plane->alb.rd=rd;
    plane->alb.rs=rs;
    plane->alb.rg=rg;
    plane->col.R=r;
    plane->col.G=g;
    plane->col.B=b;
    plane->alpha=alpha;
    plane->r_index=r_index;
    plane->shinyness=shiny;
    plane->roughness=rough;
    plane->intersect=&planeIntersect;
    plane->texImg=NULL;
    memcpy(&plane->T[0][0],&eye4x4[0][0],16*sizeof(double));
    memcpy(&plane->Tinv[0][0],&eye4x4[0][0],16*sizeof(double));
    plane->textureMap=&texMap;
    plane->frontAndBack=1;
  }
  return(plane);
}

struct object3D *newSphere(double ra, double rd, double rs, double rg, double r, double g, double b, double alpha, double r_index, double shiny, double rough) {
  // Intialize a new sphere with the specified parameters:
  // ra, rd, rs, rg - Albedos for the components of the Phong model
  // r, g, b, - Colour for this plane
  // alpha - Transparency, must be set to 1 unless you are doing refraction
  // r_index - Refraction index if you are doing refraction.
  // shiny -Exponent for the specular component of the Phong model
  //
  // This is assumed to represent a unit sphere centered at the origin.
  //

  struct object3D *sphere=(struct object3D *)calloc(1,sizeof(struct object3D));

  if (!sphere) fprintf(stderr,"Unable to allocate new sphere, out of memory!\n");
  else {
    sphere->alb.ra=ra;
    sphere->alb.rd=rd;
    sphere->alb.rs=rs;
    sphere->alb.rg=rg;
    sphere->col.R=r;
    sphere->col.G=g;
    sphere->col.B=b;
    sphere->alpha=alpha;
    sphere->r_index=r_index;
    sphere->shinyness=shiny;
    sphere->roughness=rough;
    sphere->intersect=&sphereIntersect;
    sphere->texImg=NULL;
    memcpy(&sphere->T[0][0],&eye4x4[0][0],16*sizeof(double));
    memcpy(&sphere->Tinv[0][0],&eye4x4[0][0],16*sizeof(double));
    sphere->textureMap=&texMap;
    sphere->frontAndBack=0;
  }
  return(sphere);
}

struct object3D *newCylinder(double ra, double rd, double rs, double rg, double r, double g, double b, double alpha, double r_index, double shiny, double rough) {
  // Intialize a new cylinder with the specified parameters:
  // ra, rd, rs, rg - Albedos for the components of the Phong model
  // r, g, b, - Colour for this plane
  // alpha - Transparency, must be set to 1 unless you are doing refraction
  // r_index - Refraction index if you are doing refraction.
  // shiny -Exponent for the specular component of the Phong model

  struct object3D *cylinder=(struct object3D *)calloc(1,sizeof(struct object3D));

  if (!cylinder) fprintf(stderr,"Unable to allocate new cylinder, out of memory!\n");
  else {
    cylinder->alb.ra=ra;
    cylinder->alb.rd=rd;
    cylinder->alb.rs=rs;
    cylinder->alb.rg=rg;
    cylinder->col.R=r;
    cylinder->col.G=g;
    cylinder->col.B=b;
    cylinder->alpha=alpha;
    cylinder->r_index=r_index;
    cylinder->shinyness=shiny;
    cylinder->roughness=rough;
    cylinder->intersect=&cylinderIntersect;
    cylinder->texImg=NULL;
    memcpy(&cylinder->T[0][0],&eye4x4[0][0],16*sizeof(double));
    memcpy(&cylinder->Tinv[0][0],&eye4x4[0][0],16*sizeof(double));
    cylinder->textureMap=&texMap;
    cylinder->frontAndBack=0;
  }
  return(cylinder);
}

///////////////////////////////////////////////////////////////////////////////////////
// TO DO:
//      Complete the functions that compute intersections for the canonical plane
//      and canonical sphere with a given ray. This is the most fundamental component
//      of the raytracer.
///////////////////////////////////////////////////////////////////////////////////////
void planeIntersect(struct object3D *plane, struct ray3D *ray, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b) {
  // Computes and returns the value of 'lambda' at the intersection
  // between the specified ray and the specified canonical plane.

  /////////////////////////////////
  // TO DO: Complete this function.
  /////////////////////////////////
  
  // TODO THIS INTERSECT FUNCTION CAN BE MUCH SIMPLER!
  
  // The plane is defined by the following vertices (CCW)
  // (1,1,0), (-1,1,0), (-1,-1,0), (1,-1,0)
  // With normal vector (0,0,1) (i.e. parallel to the XY plane)

  // Build a matrix to hold the linear system we want to solve.
  float A[3][3];
  A[0][0] = 1 - (-1);
  A[0][1] = 1 - (1);
  A[0][2] = ray->d.px;
  A[1][0] = 1 - (1);
  A[1][1] = 1 - (-1);
  A[1][2] = ray->d.py;
  A[2][0] = 0 - (0);
  A[2][1] = 0 - (0);
  A[2][2] = ray->d.pz;
  bool success = invert3x3Mat(A);
  if (!success) {
    *lambda = -1;
  } else {
    double beta = A[0][0] * (1 - ray->p0.px) + A[0][1] * (1 - ray->p0.py) + A[0][2] * (1 - ray->p0.pz);
    double gamma = A[1][0] * (1 - ray->p0.px) + A[1][1] * (1 - ray->p0.py) + A[1][2] * (1 - ray->p0.pz);
    double t = A[2][0] * (1 - ray->p0.px) + A[2][1] * (1 - ray->p0.py) + A[2][2] * (1 - ray->p0.pz);
    if (beta >= 0 && beta <= 1 && gamma >= 0 && gamma <= 1 && t > 0) {
      *lambda = t;
    } else {
      *lambda = -1;
    }
  }

  // If there was an intersection, compute the point of intersection and the
  // normal and the texture coordinates.
  if (lambda > 0) {
    p->px = ray->p0.px + *lambda * ray->d.px;
    p->py = ray->p0.py + *lambda * ray->d.py;
    p->pz = ray->p0.pz + *lambda * ray->d.pz;
    p->pw = 1.0;
    // For the canonical plane, the normal is always the same!
    n->px = 0.0;
    n->py = 0.0;
    n->pz = 1.0;
    n->pw = 0.0;
    // The texture coordinates are just scaled x and y coordinates. We also
    // clamp it to [0, 1] in case of rounding errors.
    *a = max(0, min(1, p->px/2 + 0.5));
    *b = max(0, min(1, p->py/2 + 0.5));
  }
}

void cylinderIntersect(struct object3D *cylinder, struct ray3D *ray, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b) {
  // Computes and returns the value of 'lambda' at the intersection between the
  // specified ray and the specified canonical cylinder. The cylinder is define
  // by two unit circles in the XY plane, at z=0 and z=1.
  *lambda = -1;

  if (ray->d.pz != 0) {
    // First check for intersection with the z=0 plane
    double lambda1  = (0 - ray->p0.pz) / ray->d.pz;
    if (lambda1 > 0) {
      double x = ray->p0.px + lambda1 * ray->d.px;
      double y = ray->p0.py + lambda1 * ray->d.py;
      // If we intersection the z=0 plane inside the unit circle, then we
      // intersect the bottom of the cylinder.
      if (x*x + y*y <= 1) {
	//printf("intersect bottom!\n");
	*lambda = lambda1;
	p->px = x;
	p->py = y;
	p->pz = 0;
	p->pw = 1;
	// Normal for the bottom of the cylinder is just the -z vector
	n->px = 0;
	n->py = 0;
	n->pz = -1;
	n->pw = 0;
      }
    }

    // Now check for intersection with the z=1 plane
    double lambda2  = (1 - ray->p0.pz) / ray->d.pz;
    if (lambda2 > 0) {
      double x = ray->p0.px + lambda2 * ray->d.px;
      double y = ray->p0.py + lambda2 * ray->d.py;
      // If we intersection the z=0 plane inside the unit circle, then we
      // intersect the bottom of the cylinder.
      if (x*x + y*y <= 1) {
	// Check if our lambda is better than the previous lambda (if any)
	if (*lambda < 0 || lambda2 < *lambda) {
	  //printf("intersect top!\n");
	  *lambda = lambda2;
	  p->px = x;
	  p->py = y;
	  p->pz = 0;
	  p->pw = 1;
	  // Normal for the top of the cylinder is just the +z vector
	  n->px = 0;
	  n->py = 0;
	  n->pz = 1;
	  n->pw = 0;
	}
      }
    }
  }

  // Finally, check for intersection with the body of the cylinder. We do this
  // by solving a quadratic to find where/whether the ray would intersect an
  // infinite cylinder (extending off to infinity in the +z and -z directions)
  // and then we check to see if the intersection point is between z=0 and z=1.
  double lambda_quad = -1;
  struct ray3D ray_xy;
  ray_xy = *ray;
  ray_xy.p0.pz = 0;
  ray_xy.d.pz = 0;
  double A = dot(&(ray_xy.d), &(ray_xy.d));
  double B = 2 * dot(&(ray_xy.d), &(ray_xy.p0));
  double C = dot(&(ray_xy.p0), &(ray_xy.p0)) - 1;
  double discriminant = B*B - 4*A*C;
  if (discriminant >= 0) {
    // One or two solutions. Find the smallest positive solution that actually
    // interesects the *finite* cylinder.
    double lambda3 = (-B - sqrt(discriminant))/(2*A);
    double lambda4 = (-B + sqrt(discriminant))/(2*A);
    double lambda3_z = ray->p0.pz + lambda3 * ray->d.pz;
    double lambda4_z = ray->p0.pz + lambda4 * ray->d.pz;
    if (lambda3 > 0 && lambda3_z >= 0 && lambda3_z <= 1) {
      lambda_quad = lambda3;
    } else if (lambda4 > 0 && lambda4_z >= 0 && lambda4_z <= 1) {
      lambda_quad = lambda4;
    }
  }

  // Now check if lambda_quad was better than any previous lambda (if any)
  if (lambda_quad > 0 && (*lambda < 0 || lambda_quad < *lambda)) {
    //printf("intersect side!\n");
    *lambda = lambda_quad;
    p->px = ray->p0.px + *lambda * ray->d.px;
    p->py = ray->p0.py + *lambda * ray->d.py;
    p->pz = ray->p0.pz + *lambda * ray->d.pz;
    p->pw = 1.0;
    // For the curved surface of the canonical cone, the normal is simply the
    // point of intersection with the z component dropped.
    *n = *p;
    n->pz = 0.0;
    n->pw = 0.0;
  }

  // Texture mapping not supported for cones
  *a = 0;
  *b = 0;
}

void sphereIntersect(struct object3D *sphere, struct ray3D *ray, double *lambda, struct point3D *p, struct point3D *n, double *a, double *b) {
  // Computes and returns the value of 'lambda' at the intersection
  // between the specified ray and the specified canonical sphere.

  /////////////////////////////////
  // TO DO: Complete this function.
  /////////////////////////////////
  double A = dot(&(ray->d), &(ray->d));
  double B = 2 * dot(&(ray->d), &(ray->p0));
  double C = dot(&(ray->p0), &(ray->p0)) - 1;
  double discriminant = B*B - 4*A*C;
  if (discriminant < 0) {
    // No solutions
    *lambda = -1;
  } else {
    // One or two solutions. Find the smallest positive solution.
    double lambda1 = (-B - sqrt(discriminant))/(2*A);
    double lambda2 = (-B + sqrt(discriminant))/(2*A);
    if (lambda1 > 0) {
      *lambda = lambda1;
    } else if (lambda2 > 0) {
      *lambda = lambda2;
    } else {
      *lambda = -1;
    }
  }

  // If there was an intersection, compute the point of intersection and the
  // normal and the texture coordinates.
  if (*lambda > 0) {
    p->px = ray->p0.px + *lambda * ray->d.px;
    p->py = ray->p0.py + *lambda * ray->d.py;
    p->pz = ray->p0.pz + *lambda * ray->d.pz;
    p->pw = 1.0;
    // For the unit sphere at the origin, the point of intersection *is also*
    // the unit normal!
    *n = *p;
    n->pw = 0.0;
    // Texture coordinates are calculated by converting intersection point to
    // sphereical cooridnates. We also clamp it to [0, 1] in case of rounding
    // errors.
    *a = max(0, min(1, p->px/2 + 0.5));
    *b = max(0, min(1, p->py/2 + 0.5));

    *a = max(0, min(1, atan2(p->py, p->px) / (2*PI) + 0.5));
    *b = max(0, min(1, atan2(sqrt(p->px*p->px + p->py*p->py), p->pz) / PI));
  }
}

void loadTexture(struct object3D *o, const char *filename) {
  // Load a texture image from file and assign it to the
  // specified object
  if (o!=NULL) {
    // We have previously loaded a texture for this object, need to de-allocate
    // it.
    if (o->texImg!=NULL) {                     
      if (o->texImg->rgbdata!=NULL) free(o->texImg->rgbdata);
      free(o->texImg);
    }
    o->texImg=readPPMimage(filename);     // Allocate new texture
  }
}

void texMap(struct image *img, double a, double b, double *R, double *G, double *B) {
  /*
    Function to determine the colour of a textured object at
    the normalized texture coordinates (a,b).

    a and b are texture coordinates in [0 1].
    img is a pointer to the image structure holding the texture for
    a given object.

    The colour is returned in R, G, B. Uses bi-linear interpolation
    to determine texture colour.
  */

  //////////////////////////////////////////////////
  // TO DO (Assignment 4 only):
  //
  //  Complete this function to return the colour
  // of the texture image at the specified texture
  // coordinates. Your code should use bi-linear
  // interpolation to obtain the texture colour.
  //////////////////////////////////////////////////
  double *rgbTexture = (double*)(img->rgbdata);
  double x = a * (img->sx-1);
  double y = b * (img->sy-1);
  int x_low = floor(x);
  int x_high = ceil(x);
  int y_low = floor(y);
  int y_high = ceil(y);
  /* printf("(a, b) = %f %f, (x, y) = %f %f, (x_low, y_low) = %d %d, (x_high, y_high) = %d %d\n", */
  /* 	 a, b, x, y, x_low, y_low, x_high, y_high); */
  // First do interpolation in the x direction
  struct colourRGB y_low_col;
  struct colourRGB y_high_col;
  if (x_low == x_high) {
    y_low_col.R = rgbTexture[3*img->sx*y_low + 3*x_low + 0];
    y_low_col.G = rgbTexture[3*img->sx*y_low + 3*x_low + 1];
    y_low_col.B = rgbTexture[3*img->sx*y_low + 3*x_low + 2];
    y_high_col.R = rgbTexture[3*img->sx*y_high + 3*x_low + 0];
    y_high_col.G = rgbTexture[3*img->sx*y_high + 3*x_low + 1];
    y_high_col.B = rgbTexture[3*img->sx*y_high + 3*x_low + 2];
  } else {
    y_low_col.R = ((x_high - x) * rgbTexture[3*img->sx*y_low + 3*x_low + 0] +
		   (x - x_low) * rgbTexture[3*img->sx*y_low + 3*x_high + 0]);
    y_low_col.G = ((x_high - x) * rgbTexture[3*img->sx*y_low + 3*x_low + 1] +
		   (x - x_low) * rgbTexture[3*img->sx*y_low + 3*x_high + 1]);
    y_low_col.B = ((x_high - x) * rgbTexture[3*img->sx*y_low + 3*x_low + 2] +
		   (x - x_low) * rgbTexture[3*img->sx*y_low + 3*x_high + 2]);
    y_high_col.R = ((x_high - x) * rgbTexture[3*img->sx*y_high + 3*x_low + 0] +
		    (x - x_low) * rgbTexture[3*img->sx*y_high + 3*x_high + 0]);
    y_high_col.G = ((x_high - x) * rgbTexture[3*img->sx*y_high + 3*x_low + 1] +
		    (x - x_low) * rgbTexture[3*img->sx*y_high + 3*x_high + 1]);
    y_high_col.B = ((x_high - x) * rgbTexture[3*img->sx*y_high + 3*x_low + 2] +
		    (x - x_low) * rgbTexture[3*img->sx*y_high + 3*x_high + 2]);
  }
  // Now do interpolation in the y direction
  if (y_low == y_high) {
    *(R) = y_low_col.R;
    *(G) = y_low_col.G;
    *(B) = y_low_col.B;
  } else {
    *(R) = (y_high - y) * y_low_col.R + (y - y_low) * y_high_col.R;
    *(G) = (y_high - y) * y_low_col.G + (y - y_low) * y_high_col.G;
    *(B) = (y_high - y) * y_low_col.B + (y - y_low) * y_high_col.B;
  }
  return;
}

void insertObject(struct object3D *o, struct object3D **list) {
  if (o==NULL) return;
  // Inserts an object into the object list.
  if (*(list)==NULL) {
    *(list)=o;
    (*(list))->next=NULL;
  } else {
    o->next=(*(list))->next;
    (*(list))->next=o;
  }
}

void insertPLS(struct pointLS *l, struct pointLS **list) {
  if (l==NULL) return;
  // Inserts a light source into the list of light sources
  if (*(list)==NULL) {
    *(list)=l;
    (*(list))->next=NULL;
  } else {
    l->next=(*(list))->next;
    (*(list))->next=l;
  }

}

void addAreaLight(float sx, float sy, float nx, float ny, float nz,\
                  float tx, float ty, float tz, int lx, int ly,\
                  float r, float g, float b, struct object3D **o_list, struct pointLS **l_list) {
  /*
    This function sets up and inserts a rectangular area light source
    with size (sx, sy)
    orientation given by the normal vector (nx, ny, nz)
    centered at (tx, ty, tz)
    consisting of (lx x ly) point light sources (uniformly sampled)
    and with colour (r,g,b) - which also determines intensity

    Note that the light source is visible as a uniformly colored rectangle and
    casts no shadow. If you require a lightsource to shade another, you must
    make it into a proper solid box with backing and sides of non-light-emitting
    material
  */

  /////////////////////////////////////////////////////
  // TO DO: (Assignment 4!)
  // Implement this function to enable area light sources
  /////////////////////////////////////////////////////

}

///////////////////////////////////
// Geometric transformation section
///////////////////////////////////

void invert(double *T, double *Tinv) {
  // Computes the inverse of transformation matrix T.
  // the result is returned in Tinv.

  float *U, *s, *V, *rv1;
  int singFlag, i;
  float T3x3[3][3],Tinv3x3[3][3];
  double tx,ty,tz;

  // Because of the fact we're using homogeneous coordinates, we must be careful how
  // we invert the transformation matrix. What we need is the inverse of the
  // 3x3 Affine transform, and -1 * the translation component. If we just invert
  // the entire matrix, junk happens.
  // So, we need a 3x3 matrix for inversion:
  T3x3[0][0]=(float)*(T+(0*4)+0);
  T3x3[0][1]=(float)*(T+(0*4)+1);
  T3x3[0][2]=(float)*(T+(0*4)+2);
  T3x3[1][0]=(float)*(T+(1*4)+0);
  T3x3[1][1]=(float)*(T+(1*4)+1);
  T3x3[1][2]=(float)*(T+(1*4)+2);
  T3x3[2][0]=(float)*(T+(2*4)+0);
  T3x3[2][1]=(float)*(T+(2*4)+1);
  T3x3[2][2]=(float)*(T+(2*4)+2);
  // Happily, we don't need to do this often.
  // Now for the translation component:
  tx=-(*(T+(0*4)+3));
  ty=-(*(T+(1*4)+3));
  tz=-(*(T+(2*4)+3));

  // Invert the affine transform
  U=NULL;
  s=NULL;
  V=NULL;
  rv1=NULL;
  singFlag=0;

  SVD(&T3x3[0][0],3,3,&U,&s,&V,&rv1);
  if (U==NULL||s==NULL||V==NULL) {
    fprintf(stderr,"Error: Matrix not invertible for this object, returning identity\n");
    memcpy(Tinv,eye4x4,16*sizeof(double));
    return;
  }

  // Check for singular matrices...
  for (i=0;i<3;i++) if (*(s+i)<1e-9) singFlag=1;
  if (singFlag) {
    fprintf(stderr,"Error: Transformation matrix is singular, returning identity\n");
    memcpy(Tinv,eye4x4,16*sizeof(double));
    return;
  }

  // Compute and store inverse matrix
  InvertMatrix(U,s,V,3,&Tinv3x3[0][0]);

  // Now stuff the transform into Tinv
  *(Tinv+(0*4)+0)=(double)Tinv3x3[0][0];
  *(Tinv+(0*4)+1)=(double)Tinv3x3[0][1];
  *(Tinv+(0*4)+2)=(double)Tinv3x3[0][2];
  *(Tinv+(1*4)+0)=(double)Tinv3x3[1][0];
  *(Tinv+(1*4)+1)=(double)Tinv3x3[1][1];
  *(Tinv+(1*4)+2)=(double)Tinv3x3[1][2];
  *(Tinv+(2*4)+0)=(double)Tinv3x3[2][0];
  *(Tinv+(2*4)+1)=(double)Tinv3x3[2][1];
  *(Tinv+(2*4)+2)=(double)Tinv3x3[2][2];
  *(Tinv+(0*4)+3)=Tinv3x3[0][0]*tx + Tinv3x3[0][1]*ty + Tinv3x3[0][2]*tz;
  *(Tinv+(1*4)+3)=Tinv3x3[1][0]*tx + Tinv3x3[1][1]*ty + Tinv3x3[1][2]*tz;
  *(Tinv+(2*4)+3)=Tinv3x3[2][0]*tx + Tinv3x3[2][1]*ty + Tinv3x3[2][2]*tz;
  *(Tinv+(3*4)+3)=1;

  free(U);
  free(s);
  free(V);
}

void RotateX(struct object3D *o, double theta) {
  // Multiply the current object transformation matrix T in object o
  // by a matrix that rotates the object theta *RADIANS* around the
  // X axis.

  double R[4][4];
  memset(&R[0][0],0,16*sizeof(double));

  R[0][0]=1.0;
  R[1][1]=cos(theta);
  R[1][2]=-sin(theta);
  R[2][1]=sin(theta);
  R[2][2]=cos(theta);
  R[3][3]=1.0;

  matMult(R,o->T);
}

void RotateY(struct object3D *o, double theta) {
  // Multiply the current object transformation matrix T in object o
  // by a matrix that rotates the object theta *RADIANS* around the
  // Y axis.

  double R[4][4];
  memset(&R[0][0],0,16*sizeof(double));

  R[0][0]=cos(theta);
  R[0][2]=sin(theta);
  R[1][1]=1.0;
  R[2][0]=-sin(theta);
  R[2][2]=cos(theta);
  R[3][3]=1.0;

  matMult(R,o->T);
}

void RotateZ(struct object3D *o, double theta) {
  // Multiply the current object transformation matrix T in object o
  // by a matrix that rotates the object theta *RADIANS* around the
  // Z axis.

  double R[4][4];
  memset(&R[0][0],0,16*sizeof(double));

  R[0][0]=cos(theta);
  R[0][1]=-sin(theta);
  R[1][0]=sin(theta);
  R[1][1]=cos(theta);
  R[2][2]=1.0;
  R[3][3]=1.0;

  matMult(R,o->T);
}

void Translate(struct object3D *o, double tx, double ty, double tz) {
  // Multiply the current object transformation matrix T in object o
  // by a matrix that translates the object by the specified amounts.

  double tr[4][4];
  memset(&tr[0][0],0,16*sizeof(double));

  tr[0][0]=1.0;
  tr[1][1]=1.0;
  tr[2][2]=1.0;
  tr[0][3]=tx;
  tr[1][3]=ty;
  tr[2][3]=tz;
  tr[3][3]=1.0;

  matMult(tr,o->T);
}

void Scale(struct object3D *o, double sx, double sy, double sz) {
  // Multiply the current object transformation matrix T in object o
  // by a matrix that scales the object as indicated.

  double S[4][4];
  memset(&S[0][0],0,16*sizeof(double));

  S[0][0]=sx;
  S[1][1]=sy;
  S[2][2]=sz;
  S[3][3]=1.0;

  matMult(S,o->T);
}

void printmatrix(double mat[4][4]) {
  fprintf(stderr,"Matrix contains:\n");
  fprintf(stderr,"%f %f %f %f\n",mat[0][0],mat[0][1],mat[0][2],mat[0][3]);
  fprintf(stderr,"%f %f %f %f\n",mat[1][0],mat[1][1],mat[1][2],mat[1][3]);
  fprintf(stderr,"%f %f %f %f\n",mat[2][0],mat[2][1],mat[2][2],mat[2][3]);
  fprintf(stderr,"%f %f %f %f\n",mat[3][0],mat[3][1],mat[3][2],mat[3][3]);
}

/////////////////////////////////////////
// Camera and view setup
/////////////////////////////////////////
struct view *setupView(struct point3D *e, struct point3D *g, struct point3D *up, double f, double wl, double wt, double wsize) {
  /*
    This function sets up the camera axes and viewing direction as discussed in the
    lecture notes.
    e - Camera center
    g - Gaze direction
    up - Up vector
    fov - Fild of view in degrees
    f - focal length
  */
  struct view *c;
  struct point3D *u, *v;

  u=v=NULL;

  // Allocate space for the camera structure
  c=(struct view *)calloc(1,sizeof(struct view));
  if (c==NULL) {
    fprintf(stderr,"Out of memory setting up camera model!\n");
    return(NULL);
  }

  // Set up camera center and axes
  c->e.px=e->px;         // Copy camera center location, note we must make sure
  c->e.py=e->py;         // the camera center provided to this function has pw=1
  c->e.pz=e->pz;
  c->e.pw=1;

  // Set up w vector (camera's Z axis). w=-g/||g||
  c->w.px=-g->px;
  c->w.py=-g->py;
  c->w.pz=-g->pz;
  c->w.pw=1;
  normalize(&c->w);

  // Set up the horizontal direction, which must be perpenticular to w and up
  u=cross(&c->w, up);
  normalize(u);
  c->u.px=u->px;
  c->u.py=u->py;
  c->u.pz=u->pz;
  c->u.pw=1;

  // Set up the remaining direction, v=(u x w)  - Mind the signs
  v=cross(&c->u, &c->w);
  normalize(v);
  c->v.px=v->px;
  c->v.py=v->py;
  c->v.pz=v->pz;
  c->v.pw=1;

  // Copy focal length and window size parameters
  c->f=f;
  c->wl=wl;
  c->wt=wt;
  c->wsize=wsize;

  // Set up coordinate conversion matrices
  // Camera2World matrix (M_cw in the notes)
  // Mind the indexing convention [row][col]
  c->C2W[0][0]=c->u.px;
  c->C2W[1][0]=c->u.py;
  c->C2W[2][0]=c->u.pz;
  c->C2W[3][0]=0;

  c->C2W[0][1]=c->v.px;
  c->C2W[1][1]=c->v.py;
  c->C2W[2][1]=c->v.pz;
  c->C2W[3][1]=0;

  c->C2W[0][2]=c->w.px;
  c->C2W[1][2]=c->w.py;
  c->C2W[2][2]=c->w.pz;
  c->C2W[3][2]=0;

  c->C2W[0][3]=c->e.px;
  c->C2W[1][3]=c->e.py;
  c->C2W[2][3]=c->e.pz;
  c->C2W[3][3]=1;

  // World2Camera matrix (M_wc in the notes)
  // Mind the indexing convention [row][col]
  c->W2C[0][0]=c->u.px;
  c->W2C[1][0]=c->v.px;
  c->W2C[2][0]=c->w.px;
  c->W2C[3][0]=0;

  c->W2C[0][1]=c->u.py;
  c->W2C[1][1]=c->v.py;
  c->W2C[2][1]=c->w.py;
  c->W2C[3][1]=0;

  c->W2C[0][2]=c->u.pz;
  c->W2C[1][2]=c->v.pz;
  c->W2C[2][2]=c->w.pz;
  c->W2C[3][2]=0;

  c->W2C[0][3]=-dot(&c->u,&c->e);
  c->W2C[1][3]=-dot(&c->v,&c->e);
  c->W2C[2][3]=-dot(&c->w,&c->e);
  c->W2C[3][3]=1;

  free(u);
  free(v);
  return(c);
}

/////////////////////////////////////////
// Image I/O section
/////////////////////////////////////////
struct image *readPPMimage(const char *filename) {
  // Reads an image from a .ppm file. A .ppm file is a very simple image representation
  // format with a text header followed by the binary RGB data at 24bits per pixel.
  // The header has the following form:
  //
  // P6
  // # One or more comment lines preceded by '#'
  // 340 200
  // 255
  //
  // The first line 'P6' is the .ppm format identifier, this is followed by one or more
  // lines with comments, typically used to inidicate which program generated the
  // .ppm file.
  // After the comments, a line with two integer values specifies the image resolution
  // as number of pixels in x and number of pixels in y.
  // The final line of the header stores the maximum value for pixels in the image,
  // usually 255.
  // After this last header line, binary data stores the RGB values for each pixel
  // in row-major order. Each pixel requires 3 bytes ordered R, G, and B.
  //
  // NOTE: Windows file handling is rather crotchetty. You may have to change the
  //       way this file is accessed if the images are being corrupted on read
  //       on Windows.
  //
  // readPPMdata converts the image colour information to floating point. This is so that
  // the texture mapping function doesn't have to do the conversion every time
  // it is asked to return the colour at a specific location.
  //

  FILE *f;
  struct image *im;
  char line[1024];
  int sizx,sizy;
  int i;
  unsigned char *tmp;
  double *fRGB;

  im=(struct image *)calloc(1,sizeof(struct image));
  if (im!=NULL) {
    im->rgbdata=NULL;
    f=fopen(filename,"rb+");
    if (f==NULL) {
      fprintf(stderr,"Unable to open file %s for reading, please check name and path\n",filename);
      free(im);
      return(NULL);
    }
    fgets(&line[0],1000,f);
    if (strcmp(&line[0],"P6\n")!=0) {
      fprintf(stderr,"Wrong file format, not a .ppm file or header end-of-line characters missing\n");
      free(im);
      fclose(f);
      return(NULL);
    }
    fprintf(stderr,"%s\n",line);
    // Skip over comments
    fgets(&line[0],511,f);
    while (line[0]=='#') {
      fprintf(stderr,"%s",line);
      fgets(&line[0],511,f);
    }
    sscanf(&line[0],"%d %d\n",&sizx,&sizy);           // Read file size
    fprintf(stderr,"nx=%d, ny=%d\n\n",sizx,sizy);
    im->sx=sizx;
    im->sy=sizy;

    fgets(&line[0],9,f);                          // Read the remaining header line
    fprintf(stderr,"%s\n",line);
    tmp=(unsigned char *)calloc(sizx*sizy*3,sizeof(unsigned char));
    fRGB=(double *)calloc(sizx*sizy*3,sizeof(double));
    if (tmp==NULL||fRGB==NULL) {
      fprintf(stderr,"Out of memory allocating space for image\n");
      free(im);
      fclose(f);
      return(NULL);
    }

    fread(tmp,sizx*sizy*3*sizeof(unsigned char),1,f);
    fclose(f);

    // Conversion to floating point
    for (i=0; i<sizx*sizy*3; i++) *(fRGB+i)=((double)*(tmp+i))/255.0;
    free(tmp);
    im->rgbdata=(void *)fRGB;

    return(im);
  }

  fprintf(stderr,"Unable to allocate memory for image structure\n");
  return(NULL);
}

struct image *newImage(int size_x, int size_y) {
  // Allocates and returns a new image with all zeros. Assumes 24 bit per pixel,
  // unsigned char array.
  struct image *im;

  im=(struct image *)calloc(1,sizeof(struct image));
  if (im!=NULL) {
    im->rgbdata=NULL;
    im->sx=size_x;
    im->sy=size_y;
    im->rgbdata=(void *)calloc(size_x*size_y*3,sizeof(unsigned char));
    if (im->rgbdata!=NULL) return(im);
  }
  fprintf(stderr,"Unable to allocate memory for new image\n");
  return(NULL);
}

void imageOutput(struct image *im, const char *filename) {
  // Writes out a .ppm file from the image data contained in 'im'.
  // Note that Windows typically doesn't know how to open .ppm
  // images. Use Gimp or any other seious image processing
  // software to display .ppm images.
  // Also, note that because of Windows file format management,
  // you may have to modify this file to get image output on
  // Windows machines to work properly.
  //
  // Assumes a 24 bit per pixel image stored as unsigned chars
  //

  FILE *f;

  if (im!=NULL)
    if (im->rgbdata!=NULL) {
      f=fopen(filename,"wb+");
      if (f==NULL) {
	fprintf(stderr,"Unable to open file %s for output! No image written\n",filename);
	return;
      }
      fprintf(f,"P6\n");
      fprintf(f,"# Output from RayTracer.c\n");
      fprintf(f,"%d %d\n",im->sx,im->sy);
      fprintf(f,"255\n");
      fwrite((unsigned char *)im->rgbdata,im->sx*im->sy*3*sizeof(unsigned char),1,f);
      fclose(f);
      return;
    }
  fprintf(stderr,"imageOutput(): Specified image is empty. Nothing output\n");
}

void deleteImage(struct image *im) {
  // De-allocates memory reserved for the image stored in 'im'
  if (im!=NULL) {
    if (im->rgbdata!=NULL) free(im->rgbdata);
    free(im);
  }
}

void cleanup(struct object3D *o_list, struct pointLS *l_list) {
  // De-allocates memory reserved for the object list and the point light source
  // list. Note that *YOU* must de-allocate any memory reserved for images
  // rendered by the raytracer.
  struct object3D *p, *q;
  struct pointLS *r, *s;

  p=o_list;              // De-allocate all memory from objects in the list
  while(p!=NULL) {
    q=p->next;
    if (p->texImg!=NULL) {
      if (p->texImg->rgbdata!=NULL) free(p->texImg->rgbdata);
      free(p->texImg);
    }
    free(p);
    p=q;
  }

  r=l_list;
  while(r!=NULL) {
    s=r->next;
    free(r);
    r=s;
  }
}
