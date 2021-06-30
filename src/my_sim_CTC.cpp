/*  
    This file is a modified version of basic.cpp,
    which was distributed as part of MuJoCo.
    Copyright (C) 2017 Roboti LLC.

    Modified by Yinjie Lin, Zhejiang University
    2021.6.30
*/


#include "mujoco.h"
#include "glfw3.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include<iostream>
#include<eigen3/Eigen/Dense>
#include<pseudo_inversion.h>

// MuJoCo data structures
mjModel* m = NULL;                  // MuJoCo model
mjData* d = NULL;                   // MuJoCo data
mjvCamera cam;                      // abstract camera
mjvOption opt;                      // visualization options
mjvScene scn;                       // abstract scene
mjrContext con;                     // custom GPU context

//init parameters
mjtNum elasped_time = 0.0;
float_t ctrl_update_freq = 1000.0;
float_t sample = 0.001;
mjtNum last_update = 0.0;
Eigen::Matrix<mjtNum, 7, 1> init_qpos;
Eigen::Matrix<mjtNum, 3, 1> init_pos;
Eigen::Quaterniond init_ori;
Eigen::Matrix<mjtNum, 6, 7> jacobian_init;
const int nsite = 2;

// mouse interaction
bool button_left = false;
bool button_middle = false;
bool button_right =  false;
double lastx = 0;
double lasty = 0;


// keyboard callback
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // backspace: reset simulation
    if( act==GLFW_PRESS && key==GLFW_KEY_BACKSPACE )
    {
        mj_resetData(m, d);
        mj_forward(m, d);
    }
}

// mouse button callback
void mouse_button(GLFWwindow* window, int button, int act, int mods)
{
    // update button state
    button_left =   (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT)==GLFW_PRESS);
    button_middle = (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE)==GLFW_PRESS);
    button_right =  (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT)==GLFW_PRESS);

    // update mouse position
    glfwGetCursorPos(window, &lastx, &lasty);
}


// mouse move callback
void mouse_move(GLFWwindow* window, double xpos, double ypos)
{
    // no buttons down: nothing to do
    if( !button_left && !button_middle && !button_right )
        return;

    // compute mouse displacement, save
    double dx = xpos - lastx;
    double dy = ypos - lasty;
    lastx = xpos;
    lasty = ypos;

    // get current window size
    int width, height;
    glfwGetWindowSize(window, &width, &height);

    // get shift key state
    bool mod_shift = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT)==GLFW_PRESS ||
                      glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT)==GLFW_PRESS);

    // determine action based on mouse button
    mjtMouse action;
    if( button_right )
        action = mod_shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    else if( button_left )
        action = mod_shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    else
        action = mjMOUSE_ZOOM;

    // move camera
    mjv_moveCamera(m, action, dx/height, dy/height, &scn, &cam);
}


// scroll callback
void scroll(GLFWwindow* window, double xoffset, double yoffset)
{
    // emulate vertical mouse motion = 5% of window height
    mjv_moveCamera(m, mjMOUSE_ZOOM, 0, -0.05*yoffset, &scn, &cam);
}

void CTC_control(const mjModel* m, mjData* d){
    //current state
    Eigen::Map<Eigen::Matrix<mjtNum, 7, 1> >q(d->qpos);
    Eigen::Map<Eigen::Matrix<mjtNum, 7, 1> >q_dot(d->qvel);
   
    //dynamics
    Eigen::Map<Eigen::Matrix<mjtNum, 7, 1> > cori(d->qfrc_bias);
    mjtNum mass_[49];
    mj_fullM(m, mass_, d->qM);
    Eigen::Map<Eigen::Matrix<mjtNum, 7, 7, Eigen::RowMajor> > mass(mass_);

    //forward kinematics
    mj_kinematics(m,d);
    int ft_site;
    ft_site = mj_name2id(m, 6, "peg_ft_site");
    mjtNum* site_xpos_;
    mjtNum* site_xmat_;
    site_xpos_ = d->site_xpos;
    site_xmat_ = d->site_xmat;
    Eigen::Map<Eigen::Matrix<mjtNum, nsite, 3, Eigen::RowMajor> > site_xpos(site_xpos_);
    Eigen::Matrix<mjtNum, 3, 1> ft_pos(site_xpos.row(ft_site));

    Eigen::Map<Eigen::Matrix<mjtNum, nsite, 9, Eigen::RowMajor> > site_xmat(site_xmat_);    
    Eigen::Map<Eigen::Matrix<mjtNum, 3, 3, Eigen::RowMajor> > ft_ori_mat(site_xmat.row(ft_site).data());
    Eigen::Quaterniond ft_ori(ft_ori_mat);
        
    //jacobian
    mjtNum jacp_[21];
    mjtNum jacr_[21];
    mj_jacSite(m, d, jacp_, jacr_, ft_site);
    Eigen::Map<Eigen::Matrix<mjtNum, 3, 7, Eigen::RowMajor> >jacp(jacp_);
    Eigen::Map<Eigen::Matrix<mjtNum, 3, 7, Eigen::RowMajor> >jacr(jacr_);
    Eigen::Matrix<mjtNum, 6, 7> jacobian;
    jacobian.block(0,0,3,7) = jacp;
    jacobian.block(3,0,3,7) = jacr;

    Eigen::Matrix<mjtNum, 7, 1> qd, qd_dot, qd_ddot, q_error, dq_error, k_gain, d_gain;
    qd_dot.setZero();
    qd_ddot.setZero();
    qd = init_qpos;

    //error
    q_error = q - qd;
    dq_error = q_dot - qd_dot;    

    //control design
    k_gain << 600.0, 600.0, 600.0, 600.0, 250.0, 250.0, 50.0;
    d_gain << 50.0, 50.0, 50.0, 50.0, 20.0, 20.0, 10.0;
    Eigen::Matrix<mjtNum, 7, 7> k_p = k_gain.asDiagonal();
    Eigen::Matrix<mjtNum, 7, 7> k_d = d_gain.asDiagonal();

    Eigen::Matrix<mjtNum, 7, 1> tau_c;
    Eigen::Matrix<mjtNum, 7, 1> tau_s;
    Eigen::Matrix<mjtNum, 7, 1> tau_d;
    // Eigen::Matrix<mjtNum, 7, 1> tau_sd;
    tau_c = mass * qd_ddot + cori;
    tau_s = -mass * (k_p * q_error + k_d * dq_error);
    tau_d = tau_c + tau_s;

    for (size_t i=0;i<7;i++)
    {
        d->ctrl[i] = tau_d[i];
    }
    
}

// main function
int main(int argc, const char** argv)
{
    // check command-line arguments
    if( argc!=2 )
    {
        printf(" USAGE:  basic modelfile\n");
        return 0;
    }

    // activate software
    mj_activate("mjkey.txt");

    // load and compile model
    char error[1000] = "Could not load binary model";
    if( strlen(argv[1])>4 && !strcmp(argv[1]+strlen(argv[1])-4, ".mjb") )
        m = mj_loadModel(argv[1], 0);
    else
        m = mj_loadXML(argv[1], 0, error, 1000);
    if( !m )
        mju_error_s("Load model error: %s", error);

    // make data
    d = mj_makeData(m);

    // init GLFW
    if( !glfwInit() )
        mju_error("Could not initialize GLFW");

    // create window, make OpenGL context current, request v-sync
    GLFWwindow* window = glfwCreateWindow(1200, 900, "Demo", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    // initialize visualization data structures
    mjv_defaultCamera(&cam);
    mjv_defaultOption(&opt);
    mjv_defaultScene(&scn);
    mjr_defaultContext(&con);

    // create scene and context
    mjv_makeScene(m, &scn, 2000);
    mjr_makeContext(m, &con, mjFONTSCALE_150);

    // install GLFW mouse and keyboard callbacks
    glfwSetKeyCallback(window, keyboard);
    glfwSetCursorPosCallback(window, mouse_move);
    glfwSetMouseButtonCallback(window, mouse_button);
    glfwSetScrollCallback(window, scroll);

    //init position
    // init_pos << 0, M_PI / 16, 0, -M_PI / 2 - M_PI / 3, 0, M_PI - 0.2, -M_PI/4;
    init_qpos << 0.6579, -0.06562,-0.6566,-2.339,-0.0441,2.2911,0.7280;
    for (size_t i=0;i<7;i++)
    {
    d->qpos[i] = init_qpos[i];
    }
    
    // get init pos and ori
    mj_kinematics(m,d);
    int ft_site;
    // nsite = m->nsite;
    ft_site = mj_name2id(m, 6, "peg_ft_site");
    mjtNum* site_xpos_;
    mjtNum* site_xmat_;
    site_xpos_ = d->site_xpos;
    site_xmat_ = d->site_xmat;
    Eigen::Map<Eigen::Matrix<mjtNum, nsite, 3, Eigen::RowMajor> > site_xpos(site_xpos_);
    Eigen::Matrix<mjtNum, 3, 1> init_pos(site_xpos.row(ft_site));

    Eigen::Map<Eigen::Matrix<mjtNum, nsite, 9, Eigen::RowMajor> > site_xmat(site_xmat_);    
    Eigen::Map<Eigen::Matrix<mjtNum, 3, 3, Eigen::RowMajor> > ft_ori_mat(site_xmat.row(ft_site).data());
    Eigen::Quaterniond init_ori(ft_ori_mat);

    //get init jacobian
    mjtNum jacp_[21];
    mjtNum jacr_[21];
    mj_jacSite(m, d, jacp_, jacr_, ft_site);
    Eigen::Map<Eigen::Matrix<mjtNum, 3, 7, Eigen::RowMajor> >jacp(jacp_);
    Eigen::Map<Eigen::Matrix<mjtNum, 3, 7, Eigen::RowMajor> >jacr(jacr_);
    jacobian_init.block(0,0,3,7) = jacp;
    jacobian_init.block(3,0,3,7) = jacr;

    //define the control callback
    mjcb_control = CTC_control;
    // run main loop, target real-time simulation and 60 fps rendering
    while( !glfwWindowShouldClose(window) )
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        mjtNum simstart = d->time;
        while( d->time - simstart < 1.0/60.0 ) 
        {
        mj_step(m, d);
        }

        // get framebuffer viewport
        mjrRect viewport = {0, 0, 0, 0};
        glfwGetFramebufferSize(window, &viewport.width, &viewport.height);

        // update scene and render
        mjv_updateScene(m, d, &opt, NULL, &cam, mjCAT_ALL, &scn);
        mjr_render(viewport, &scn, &con);

        // swap OpenGL buffers (blocking call due to v-sync)
        glfwSwapBuffers(window);

        // process pending GUI events, call GLFW callbacks
        glfwPollEvents();
    }

    //free visualization storage
    mjv_freeScene(&scn);
    mjr_freeContext(&con);

    // free MuJoCo model and data, deactivate
    mj_deleteData(d);
    mj_deleteModel(m);
    mj_deactivate();

    // terminate GLFW (crashes with Linux NVidia drivers)
    #if defined(__APPLE__) || defined(_WIN32)
        glfwTerminate();
    #endif

    return 1;
}
