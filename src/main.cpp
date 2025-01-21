// ----------------------------------------------------------------------------
// main.cpp
//
//  Created on: Fri Jan 22 20:45:07 2021
//      Author: Kiwon Um
//        Mail: kiwon.um@telecom-paris.fr
//
// Description: SPH simulator (DO NOT DISTRIBUTE!)
//
// Copyright 2021-2024 Kiwon Um
//
// The copyright to the computer program(s) herein is the property of Kiwon Um,
// Telecom Paris, France. The program(s) may be used and/or copied only with
// the written permission of Kiwon Um or in accordance with the terms and
// conditions stipulated in the agreement/contract under which the program(s)
// have been supplied.
// ----------------------------------------------------------------------------

#include <algorithm>
#include <unistd.h>
#define _USE_MATH_DEFINES

#include <glad/gl.h>
#include <GLFW/glfw3.h>

#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>

#ifndef M_PI
#define M_PI 3.141592
#endif

#include "Vector.hpp"

// window parameters
GLFWwindow *gWindow = nullptr;
int gWindowWidth = 1024;
int gWindowHeight = 768;

// timer
float gAppTimer = 0.0;
float gAppTimerLastClockTime;
bool gAppTimerStoppedP = true;

// global options
bool gPause = true;
bool gSaveFile = false;
bool gRecord = false;
bool gShowGrid = true;
bool gShowVel = false;
int gSavedCnt = 0;

const int kViewScale = 15;

// SPH Kernel function: cubic spline
class CubicSpline {
public:
  explicit CubicSpline(const Real h=1) : _dim(2)
  {
    setSmoothingLen(h);
  }
  void setSmoothingLen(const Real h)
  {
    const Real h2 = square(h), h3 = h2*h;
    _h = h;
    _sr = 2e0*h;
    _c[0]  = 2e0/(3e0*h);
    _c[1]  = 10e0/(7e0*M_PI*h2);
    _c[2]  = 1e0/(M_PI*h3);
    _gc[0] = _c[0]/h;
    _gc[1] = _c[1]/h;
    _gc[2] = _c[2]/h;
  }
  Real smoothingLen() const { return _h; }
  Real supportRadius() const { return _sr; }

  Real f(const Real l) const
  {
    const Real q = l/_h;
    if(q<1e0) return _c[_dim-1]*(1e0 - 1.5*square(q) + 0.75*cube(q));
    else if(q<2e0) return _c[_dim-1]*(0.25*cube(2e0-q));
    return 0;
  }
  Real derivative_f(const Real l) const
  {
    const Real q = l/_h;
    if(q<=1e0) return _gc[_dim-1]*(-3e0*q+2.25*square(q));
    else if(q<2e0) return -_gc[_dim-1]*0.75*square(2e0-q);
    return 0;
  }

  Real w(const Vec2f &rij) const { return f(rij.length()); }
  Vec2f grad_w(const Vec2f &rij) const { return grad_w(rij, rij.length()); }
  Vec2f grad_w(const Vec2f &rij, const Real len) const
  {
    return derivative_f(len)*rij/len;
  }

private:
  unsigned int _dim;
  Real _h, _sr, _c[3], _gc[3];
};

class SphSolver {
public:
  explicit SphSolver(
    const Real nu=0.08, const Real h=0.5, const Real density=1e3,
    const Vec2f g=Vec2f(0, -9.8), const Real eta=0.01, const Real gamma=7.0) :
    _kernel(h), _nu(nu), _h(h), _d0(density),
    _g(g), _eta(eta), _gamma(gamma)
  {
    _dt = 0.0005;
    _m0 = _d0*_h*_h;
    _c = std::fabs(_g.y)/_eta;
    _k = _d0*_c*_c/_gamma;
  }

  // assume an arbitrary grid with the size of res_x*res_y; a fluid mass fill up
  // the size of f_width, f_height; each cell is sampled with 2x2 particles.
  void initScene(
    const int res_x, const int res_y, const int f_width, const int f_height)
  {
    _pos.clear();

    _resX = res_x;
    _resY = res_y;

    // set wall for boundary
    _l = 2.5*_h;
    _r = static_cast<Real>(res_x) - 2.5*_h;
    _b = 2.5*_h;
    _t = static_cast<Real>(res_y) - 0.5*_h;

    initBoundaryParticles();
    // sample a fluid mass
    const Vec2f fluidOffset(0.5, 3.0);  // x=1.5 (1+0.5), y=3.0 start position
    for(int j=0; j<f_height; ++j) {
      for(int i=0; i<f_width; ++i) {
        Vec2f basePos(i+1 + fluidOffset.x, j + fluidOffset.y);  // i+1 because we start at x=1
        _pos.push_back(basePos + Vec2f(0.25, 0.25));
        _pos.push_back(basePos + Vec2f(0.75, 0.25));
        _pos.push_back(basePos + Vec2f(0.25, 0.75));
        _pos.push_back(basePos + Vec2f(0.75, 0.75));
      }
    }

    // make sure for the other particle quantities
    _vel = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
    _acc = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
    _pressure   = std::vector<Real>(_pos.size(), 0);
    _density   = std::vector<Real>(_pos.size(), 0);

    _col = std::vector<float>(_pos.size()*4, 1.0); // RGBA
    _vln = std::vector<float>(_pos.size()*4, 0.0); // GL_LINES

    updateColor();
  }

  void initBoundaryParticles() {
      Real particleSpacing = _h;
      _numBoundaryParticles = 0; // Initialize counter

      // Add bottom wall
      for (Real x = 0; x < _resX; x += 1.0) {
        _pos.push_back(Vec2f(x + 0.25, 0.25));
        _pos.push_back(Vec2f(x + 0.75, 0.25));
        _pos.push_back(Vec2f(x + 0.25, 0.75));
        _pos.push_back(Vec2f(x + 0.75, 0.75));
        _numBoundaryParticles += 4;
      }

      // For side walls
      for (Real y = 1.0; y < _resY; y += 1.0) {// start at 1 because we want to avoid overlap in the corners
        // Left wall
        _pos.push_back(Vec2f(0.25, y + 0.25));
        _pos.push_back(Vec2f(0.75, y + 0.25));
        _pos.push_back(Vec2f(0.25, y + 0.75));
        _pos.push_back(Vec2f(0.75, y + 0.75));

        // Right wall
        _pos.push_back(Vec2f(_resX - 0.75, y + 0.25));
        _pos.push_back(Vec2f(_resX - 0.25, y + 0.25));
        _pos.push_back(Vec2f(_resX - 0.75, y + 0.75));
        _pos.push_back(Vec2f(_resX - 0.25, y + 0.75));
        _numBoundaryParticles+=8;
      }

      // Initialize arrays for boundary particles
      _vel.resize(_numBoundaryParticles, Vec2f(0, 0));
      _acc.resize(_numBoundaryParticles, Vec2f(0, 0));
      _pressure.resize(_numBoundaryParticles, 0);
      _density.resize(_numBoundaryParticles, 0);
      _col.resize(_numBoundaryParticles * 4, 1.0);
      _vln.resize(_numBoundaryParticles * 4, 0.0);
  }
  
  void update()
  {
    std::cout << '.' << std::flush;

    buildNeighbor();
    computeDensity();
    computePressure();

    _acc = std::vector<Vec2f>(_pos.size(), Vec2f(0, 0));
    applyBodyForce();
    applyPressureForce();
    applyViscousForce();

    updateVelocity();
    updatePosition();

    resolveCollision();

    updateColor();
    if(gShowVel) updateVelLine();
  }

  tIndex particleCount() const { return _pos.size(); }
  const Vec2f& position(const tIndex i) const { return _pos[i]; }
  const float& color(const tIndex i) const { return _col[i]; }
  const float& vline(const tIndex i) const { return _vln[i]; }

  int resX() const { return _resX; }
  int resY() const { return _resY; }

  Real equationOfState(
    const Real d, const Real d0,
    const Real k,               // NOTE: You can use _k for k here.
    const Real gamma=7.0)
  {
    // TODO: pressure calculation
    return k * (pow((d/d0), 7) -1);
  }

private:
  void getGridPos(const Vec2f& pos, int& grid_x, int& grid_y) {
    grid_x = static_cast<int>(std::floor(pos[0]));
    grid_y = static_cast<int>(std::floor(pos[1]));
    grid_x = std::max(0, std::min(grid_x, resX() - 1));
    grid_y = std::max(0, std::min(grid_y, resY() - 1));
}
  void buildNeighbor()
    {
      _pidxInGrid.clear();
      _pidxInGrid.resize(resX() * resY());
      for(int i = 0; i < _pos.size(); ++i){
        int grid_x, grid_y;
        getGridPos(_pos[i], grid_x, grid_y);
        //std::cout << "Adding neigh at pos:" << grid_x << "," << grid_y << " from pos:" << _pos[i][0] << "," << _pos[i][1] << std::endl;
        _pidxInGrid[idx1d(grid_x, grid_y)].push_back(i); 
      } 
    }

  void computeDensity() {
    // Reset boundary particle density
    for (int i = 0; i < _numBoundaryParticles; ++i) {
      _density[i] = _d0;
    }
    // Calculate density for all particles
    #pragma omp parallel for
    for (int i = _numBoundaryParticles; i < _pos.size(); ++i) {
      int grid_x, grid_y;
      getGridPos(_pos[i], grid_x, grid_y);
      _density[i] = 0;

      for(int x = std::max(grid_x-1, 0); x <= std::min(grid_x+1, resX()-1); x++) {
        for(int y = std::max(grid_y-1, 0); y <= std::min(grid_y+1, resY()-1); y++) {
          for(int neigh_particle : _pidxInGrid[idx1d(x, y)]) {
            float dist = _pos[i].distanceTo(_pos[neigh_particle]);
            _density[i] += _m0 * _kernel.f(dist);
          }
        }
      }
    }
  }

  void computePressure()
  {
    #pragma omp parallel for
    for(int i = 0; i < _pressure.size(); ++i){
      _pressure[i] = equationOfState(_density[i], _d0, _k);
      _pressure[i] = std::max(_pressure[i], 0.0f);
      //std::cout << _pressure[i] << " pressure for particle:" << i << std::endl;
    }
  }

  void applyBodyForce()
  {
    #pragma omp parallel for
    for(int i = 0; i < _acc.size(); ++i){
      _acc[i] += _m0 * _g;

    }
  }

  void applyPressureForce() {
    #pragma omp parallel for
    for(int i = 0; i < _pos.size(); ++i) {
      Vec2f pressure_force = Vec2f(0,0);
      int grid_x, grid_y;
      getGridPos(_pos[i], grid_x, grid_y);

      for(int x = std::max(grid_x-1, 0); x <= std::min(grid_x+1, resX()-1); x++) {
        for(int y = std::max(grid_y-1, 0); y <= std::min(grid_y+1, resY()-1); y++) {
          for(int neigh_particle : _pidxInGrid[idx1d(x, y)]) {
            if (i == neigh_particle) continue;

            Vec2f rij = _pos[i] - _pos[neigh_particle];
            float len = rij.length();

            // Safety check for tiny distances
            if (len < 1e-10) {
              continue;  // Skip extremely close particles
            }

            float density_i = std::max(_density[i], 1e-6f);
            float density_j = std::max(_density[neigh_particle], 1e-6f);

            float pressure_term = (_pressure[i] / (density_i * density_i) +
                                 _pressure[neigh_particle] / (density_j * density_j));

            pressure_force += (-_m0) * _m0 * pressure_term * _kernel.grad_w(rij, len);
          }
        }
      }
      _acc[i] += pressure_force/_density[i];
    }
  }

  void applyViscousForce() {
    #pragma omp parallel for
    for(int i = 0; i < _pos.size(); ++i) {
      Vec2f viscous_force = Vec2f(0, 0);
      int grid_x, grid_y;
      getGridPos(_pos[i], grid_x, grid_y);

      for(int x = std::max(grid_x-1, 0); x <= std::min(grid_x+1, resX()-1); x++) {
        for(int y = std::max(grid_y-1, 0); y <= std::min(grid_y+1, resY()-1); y++) {
          for(int neigh_particle : _pidxInGrid[idx1d(x, y)]) {
            if (i == neigh_particle) continue;

            Vec2f xij = _pos[i] - _pos[neigh_particle];
            Vec2f vij = _vel[i] - _vel[neigh_particle];
            float len = xij.length();
            
            if (len > 1e-10) { // same safety check as or pressure
              float dot_product = xij.dotProduct(vij);
              viscous_force += 2 * _nu * _m0 * (dot_product / (len * len + 0.01f * _h * _h)) 
                              * _kernel.grad_w(xij, len) / _density[neigh_particle];
            }
          }
        }
      }
      _acc[i] += viscous_force;
    }
  }

  void updateVelocity()
  {
    #pragma omp parallel for
    for(int i = _numBoundaryParticles; i < _pos.size(); ++i){
      _vel[i] += _acc[i] * _dt;
    }
  }

  void updatePosition()
  {
    #pragma omp parallel for
    for(int i = _numBoundaryParticles; i < _pos.size(); ++i){
      _pos[i] += _vel[i] * _dt;
    }
  }

  // simple collision detection/resolution for each particle
  void resolveCollision()
  {
    std::vector<tIndex> need_res;
    for(tIndex i=_numBoundaryParticles; i<particleCount(); ++i) {
      if(_pos[i].x<_l || _pos[i].y<_b || _pos[i].x>_r || _pos[i].y>_t)
        need_res.push_back(i);
    }

    for(
      std::vector<tIndex>::const_iterator it=need_res.begin();
      it<need_res.end();
      ++it) {
      const Vec2f p0 = _pos[*it];
      _pos[*it].x = clamp(_pos[*it].x, _l, _r);
      _pos[*it].y = clamp(_pos[*it].y, _b, _t);
      _vel[*it] = (_pos[*it] - p0)/_dt;
    }
  }
  
  void updateColor()
  {

    for(tIndex i=0; i<_numBoundaryParticles; ++i) {
      _col[i*4+0] = 0.6;
      _col[i*4+1] = 0.6;
      _col[i*4+2] = 0.6;
    }
    const float minDensityRatio = 0.7f;   // Below this will be lightest blue
    const float maxDensityRatio = 1.3f;   // Above this will be darkest blue
    for(tIndex i=_numBoundaryParticles; i<particleCount(); ++i) {
      float densityRatio = _density[i] / _d0;
      densityRatio = std::max(minDensityRatio, std::min(densityRatio, maxDensityRatio));
      float t = (densityRatio - minDensityRatio) / (maxDensityRatio - minDensityRatio);
      // Red component: 0.9 -> 0.0
      _col[i*4+0] = 0.9f * (1.0f - t);
      
      // Green component: 0.9 -> 0.2
      _col[i*4+1] = 0.9f * (1.0f - t) + 0.2f * t;
      
      // Blue component: always 1.0
      _col[i*4+2] = 1.0f;
    }
  }

  void updateVelLine()
  {
    for(tIndex i=0; i<particleCount(); ++i) {
      _vln[i*4+0] = _pos[i].x;
      _vln[i*4+1] = _pos[i].y;
      _vln[i*4+2] = _pos[i].x + _vel[i].x;
      _vln[i*4+3] = _pos[i].y + _vel[i].y;
    }
  }

  inline tIndex idx1d(const int i, const int j) { return i + j*resX(); }

  const CubicSpline _kernel;

  // particle data
  tIndex _numBoundaryParticles;
  std::vector<Vec2f> _pos;      // position
  std::vector<Vec2f> _vel;      // velocity
  std::vector<Vec2f> _acc;      // acceleration
  std::vector<Real>  _pressure;        // pressure
  std::vector<Real>  _density;        // density

  std::vector< std::vector<tIndex> > _pidxInGrid; // will help you find neighbor particles

  std::vector<float> _col;    // particle color; just for visualization
  std::vector<float> _vln;    // particle velocity lines; just for visualization

  // simulation
  Real _dt;                     // time step

  int _resX, _resY;             // background grid resolution

  // wall
  Real _l, _r, _b, _t;          // wall (boundary)

  // SPH coefficients
  Real _nu;                     // viscosity coefficient
  Real _d0;                     // rest density
  Real _h;                      // particle spacing (i.e., diameter)
  Vec2f _g;                     // gravity

  Real _m0;                     // rest mass
  Real _k;                      // EOS coefficient

  Real _eta;
  Real _c;                      // speed of sound
  Real _gamma;                  // EOS power factor
};

SphSolver gSolver(8.0, 0.5, 1e3, Vec2f(0, -9.8), 0.01, 7.0);

void printHelp()
{
  std::cout <<
    "> Help:" << std::endl <<
    "    Keyboard commands:" << std::endl <<
    "    * H: print this help" << std::endl <<
    "    * P: toggle simulation" << std::endl <<
    "    * G: toggle grid rendering" << std::endl <<
    "    * V: toggle velocity rendering" << std::endl <<
    "    * S: save current frame into a file" << std::endl <<
    "    * R: toggle record mode" << std::endl <<
    "    * Q: quit the program" << std::endl;
}

// Executed each time the window is resized. Adjust the aspect ratio and the rendering viewport to the current window.
void windowSizeCallback(GLFWwindow *window, int width, int height)
{
  gWindowWidth = width;
  gWindowHeight = height;
  glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

// Executed each time a key is entered.
void keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
  if(action == GLFW_PRESS && key == GLFW_KEY_H) {
    printHelp();
  } else if(action == GLFW_PRESS && key == GLFW_KEY_S) {
    gSaveFile = !gSaveFile;
  }
  if (action == GLFW_PRESS && key == GLFW_KEY_R) {
    gRecord = !gRecord;
    if (gRecord) {
      std::cout << "Recording started. Frames will be saved to current directory."<< std::endl;
    } else {
      std::cout << "Recording stopped." << std::endl;
    }
  } else if(action == GLFW_PRESS && key == GLFW_KEY_G) {
    gShowGrid = !gShowGrid;
  } else if(action == GLFW_PRESS && key == GLFW_KEY_V) {
    gShowVel = !gShowVel;
  } else if(action == GLFW_PRESS && key == GLFW_KEY_P) {
    gAppTimerStoppedP = !gAppTimerStoppedP;
    if(!gAppTimerStoppedP)
      gAppTimerLastClockTime = static_cast<float>(glfwGetTime());
  } else if(action == GLFW_PRESS && key == GLFW_KEY_Q) {
    glfwSetWindowShouldClose(window, true);
  }
}

void initGLFW()
{
  // Initialize GLFW, the library responsible for window management
  if(!glfwInit()) {
    std::cerr << "ERROR: Failed to init GLFW" << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // Before creating the window, set some option flags
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
  // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // only if requesting 3.0 or above
  // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_ANY_PROFILE); // for OpenGL below 3.2
  glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);

  // Create the window
  gWindowWidth = gSolver.resX()*kViewScale;
  gWindowHeight = gSolver.resY()*kViewScale;
  gWindow = glfwCreateWindow(
    gSolver.resX()*kViewScale, gSolver.resY()*kViewScale,
    "Basic SPH Simulator", nullptr, nullptr);
  if(!gWindow) {
    std::cerr << "ERROR: Failed to open window" << std::endl;
    glfwTerminate();
    std::exit(EXIT_FAILURE);
  }

  // Load the OpenGL context in the GLFW window
  glfwMakeContextCurrent(gWindow);

  // not mandatory for all, but MacOS X
  glfwGetFramebufferSize(gWindow, &gWindowWidth, &gWindowHeight);

  // Connect the callbacks for interactive control
  glfwSetWindowSizeCallback(gWindow, windowSizeCallback);
  glfwSetKeyCallback(gWindow, keyCallback);

  std::cout << "Window created: " <<
    gWindowWidth << ", " << gWindowHeight << std::endl;
}

void clear();

void exitOnCriticalError(const std::string &message)
{
  std::cerr << "> [Critical error]" << message << std::endl;
  std::cerr << "> [Clearing resources]" << std::endl;
  clear();
  std::cerr << "> [Exit]" << std::endl;
  std::exit(EXIT_FAILURE);
}

void initOpenGL()
{
  // Load extensions for modern OpenGL
  if(!gladLoadGL(glfwGetProcAddress))
    exitOnCriticalError("[Failed to initialize OpenGL context]");

  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);

  glViewport(0, 0, static_cast<GLint>(gWindowWidth), static_cast<GLint>(gWindowHeight));
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, gSolver.resX(), 0, gSolver.resY(), 0, 1);
}

void init()
{
  gSolver.initScene(48, 32, 16, 16);

  initGLFW();                   // Windowing system
  initOpenGL();
}

void clear()
{
  glfwDestroyWindow(gWindow);
  glfwTerminate();
}

// The main rendering call
void render()
{
  glClearColor(.4f, .4f, .4f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // grid guides
  if(gShowGrid) {
    glBegin(GL_LINES);
    for(int i=1; i<gSolver.resX(); ++i) {
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(static_cast<Real>(i), 0.0);
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(static_cast<Real>(i), static_cast<Real>(gSolver.resY()));
    }
    for(int j=1; j<gSolver.resY(); ++j) {
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(0.0, static_cast<Real>(j));
      glColor3f(0.3, 0.3, 0.3);
      glVertex2f(static_cast<Real>(gSolver.resX()), static_cast<Real>(j));
    }
    glEnd();
  }

  // render particles
  glEnable(GL_POINT_SMOOTH);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);

  glPointSize(0.5f*kViewScale);

  glColorPointer(4, GL_FLOAT, 0, &gSolver.color(0));
  glVertexPointer(2, GL_FLOAT, 0, &gSolver.position(0));
  glDrawArrays(GL_POINTS, 0, gSolver.particleCount());

  glDisableClientState(GL_COLOR_ARRAY);
  glDisableClientState(GL_VERTEX_ARRAY);

  // velocity
  if(gShowVel) {
    glColor4f(0.0f, 0.0f, 0.5f, 0.2f);

    glEnableClientState(GL_VERTEX_ARRAY);

    glVertexPointer(2, GL_FLOAT, 0, &gSolver.vline(0));
    glDrawArrays(GL_LINES, 0, gSolver.particleCount()*2);

    glDisableClientState(GL_VERTEX_ARRAY);
  }

  if(gSaveFile || gRecord) {
    std::stringstream fpath;

    if (gSaveFile) {
      fpath << "s" << std::setw(4) << std::setfill('0') << gSavedCnt++ << ".tga";
      gSaveFile = false;
    } else {
      fpath << "f" << std::setw(6) << std::setfill('0') << gSavedCnt++ << ".tga";
    }

    std::cout << "Saving file " << fpath.str() << " ... " << std::flush;
    const short int w = gWindowWidth;
    const short int h = gWindowHeight;
    std::vector<int> buf(w*h*3, 0);
    glReadPixels(0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, &(buf[0]));

    FILE *out = fopen(fpath.str().c_str(), "wb");
    short TGAhead[] = {0, 2, 0, 0, 0, 0, w, h, 24};
    fwrite(&TGAhead, sizeof(TGAhead), 1, out);
    fwrite(&(buf[0]), 3*w*h, 1, out);
    fclose(out);
    gSaveFile = false;

    std::cout << "Done" << std::endl;
  }
}

// Update any accessible variable based on the current time
void update(const float currentTime)
{
  if(!gAppTimerStoppedP) {
    // NOTE: When you want to use application's dt ...
    // const float dt = currentTime - gAppTimerLastClockTime;
    // gAppTimerLastClockTime = currentTime;
    // gAppTimer += dt;

    // solve 10 steps
    for(int i=0; i<10; ++i) gSolver.update();
  }
}

int main(int argc, char **argv)
{
  init();
  while(!glfwWindowShouldClose(gWindow)) {
    update(static_cast<float>(glfwGetTime()));
    render();
    glfwSwapBuffers(gWindow);
    glfwPollEvents();
  }
  clear();
  std::cout << " > Quit" << std::endl;
  return EXIT_SUCCESS;
}
