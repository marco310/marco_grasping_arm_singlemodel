/*
  This is the code for static grasping project
  This code is coded by Fan Yang
  Start from July, 25th 2019
*/
#include <iostream>
#include <string>
#include <signal.h>

#include "Sai2Model.h"
#include "redis/RedisClient.h"
#include "timer/LoopTimer.h"
#include "tasks/JointTask.h"
#include "tasks/PosOriTask.h"
#include "tasks/PositionTask.h"
#include "force_sensor/ForceSensorSim.h"
#include "force_sensor/ForceSensorDisplay.h"

bool runloop = false;
void sighandler(int sig)
{ runloop = false; }

using namespace std;
using namespace Eigen;

//const string robot_file = "./resources/panda_arm_hand.urdf";
//const string robot_name = "panda_arm_hand";
const string robot_file = "./resources/panda_arm_hand.urdf";
const string robot_name = "panda_arm_hand";


const std::string JOINT_ANGLES_KEY  = "sai2::grasp_marco::sensors::q";
const std::string JOINT_VELOCITIES_KEY = "sai2::grasp_marco::sensors::dq";

const std::string JOINT_TORQUES_COMMANDED_KEY = "sai2::grasp_marco::actuators::fgc";

#define NUM_OF_FINGERS_IN_MODEL 4
#define NUM_OF_FINGERS_USED     3

#define CONTACT_COEFFICIENT     0.5 
#define MIN_COLLISION_V         0.01
#define DISPLACEMENT_DIS        0.02  // how much you wanna move awat from the original point in normal detection step
#define FRICTION_COEFFICIENT     0.5

#define STREACH_HAND			0
#define REACH_POSE              1
#define PRE_GRASP	            2
#define APPROACH				3
#define CENTER_THUMB			4
#define LIFT					5

int state = STREACH_HAND;

double prob_distance = 0.006; // how much you want the to prob laterally in normal detection step

// the function used in the finger position control command
VectorXd compute_position_cmd_torques(Sai2Model::Sai2Model* robot, string link, Vector3d pos_in_link, Vector3d desired_position, double kp);
// the function used in the finger force control, used to achieve compliance
VectorXd compute_force_cmd_torques(Sai2Model::Sai2Model* robot, string link, Vector3d pos_in_link, Vector3d desired_position, double force_requeired);
// pd controller to compute the torque to get back to the specific configuration
VectorXd compute_joint_cmd_torques(Sai2Model::Sai2Model* robot, VectorXd desired_joint_angles);
// pd controller to compute the torque to get back to the specific configuration with custom kp, kv=kp/5
VectorXd compute_joint_cmd_torques(Sai2Model::Sai2Model* robot, VectorXd desired_joint_angles, double kp);
// pd controller to compute the torque to get back to the specific configuration for only a specific finger
VectorXd compute_joint_cmd_torques_one_finger(Sai2Model::Sai2Model* robot, VectorXd desired_joint_angles, int index);
// this function is used to check whether we can only 2 finger to have a leagal grasp
bool check_2_finger_grasp(vector<Vector3d> contact_points,vector<Vector3d> normals, double friction_coefficient);
bool check_3_finger_grasp(vector<Vector3d> contact_points,vector<Vector3d> normals, double friction_coefficient);
// verify whether these 3 normals can positively span the space. it's not very restrict. Because 
// this code doesn't project all the normals in the same plane
// if the normals positivelt span the space, then the cross product for an arbitrary normal with the other 2 normals
// should be one positive and one negative.
bool check_positive_span_space(vector<Vector3d> normals);
  

int main (int argc, char** argv) 
{
	auto redis_client = RedisClient();
	redis_client.connect();

	// set up signal handler
	signal(SIGABRT, &sighandler);
	signal(SIGTERM, &sighandler);
	signal(SIGINT, &sighandler);

	
	auto robot = new Sai2Model::Sai2Model(robot_file, false);

	// read from Redis
	robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
	robot->_dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_KEY);
	robot->updateModel();
	int dof = robot->dof();


	//------------------------------------------------------------
	// Robot variables
	//------------------------------------------------------------

	VectorXd command_torques = VectorXd::Zero(dof);
	VectorXd joint_torques = VectorXd::Zero(dof);
	VectorXd gravity_torques = VectorXd::Zero(dof);
	VectorXd palm_command_torques = VectorXd::Zero(dof);

	VectorXd coriolis = VectorXd::Zero(dof);
	MatrixXd N_prec = MatrixXd::Identity(dof,dof);

	//------------------------------------------------------------
	// Hand variables
	//------------------------------------------------------------

	vector<Vector3d> current_finger_position;
	VectorXd pregrasp_angles = VectorXd::Zero(dof);
	
	vector<VectorXd> finger_command_torques;
	finger_command_torques.push_back(VectorXd::Zero(dof));
	finger_command_torques.push_back(VectorXd::Zero(dof));
	finger_command_torques.push_back(VectorXd::Zero(dof));
	finger_command_torques.push_back(VectorXd::Zero(dof));
	vector<VectorXd> temp_finger_command_torques = finger_command_torques; // the raw command torques before blocking
	//control 4 fingers, finger 0,1,2,3

	// the vector used to record the velocity in the finger move close state
	vector<deque<double>> velocity_record;
	vector<deque<double>> detect_velocity_record;	

	// vector<Sai2Primitives::PositionTask *>  position_tasks;
	vector<int> detect_states;
	vector<vector<Vector3d>> contact_points;
	vector<Vector3d> normals;

	for(int i = 0; i < NUM_OF_FINGERS_USED; i++)
	{
		deque<double> temp_queue;
		temp_queue.push_back(0.0);
		temp_queue.push_back(0.0);
		velocity_record.push_back(temp_queue);
		detect_velocity_record.push_back(temp_queue);
		current_finger_position.push_back(Vector3d::Zero());
		detect_states.push_back(0);
	}

	vector<int> finger_contact_flag; // finger0, 1, 2, 3
	for (int i = 0; i < NUM_OF_FINGERS_IN_MODEL; i++)
	{
		finger_contact_flag.push_back(0);
	}


	//----------------------------------------------------------
	// Hand geometry and links
	//----------------------------------------------------------

	vector<Affine3d> poses;
	Affine3d identity_pose = Affine3d::Identity();
	Affine3d temp_pose = Affine3d::Identity();
	temp_pose.translation() = Vector3d(0.0, 0.0, 0.0437); //437
	poses.push_back(temp_pose);
	poses.push_back(temp_pose);
	poses.push_back(temp_pose);
	temp_pose.translation() = Vector3d(0.0,0.0,0.0593);
	poses.push_back(temp_pose);



	//----------------------------------------------------------
	// Object information
	//----------------------------------------------------------

	Vector3d CoM_of_object = Vector3d(0.05,0.0,0.05); // in the world frame
	CoM_of_object -= Vector3d(0.0, 0.0, 0.25); // transform into the robor frame //TODO:use the same method used in posoritask


	//-------------------------------------------------------------
	//Posori information
	//-------------------------------------------------------------
	string link_name = "link7";
	Eigen::Vector3d pos_in_link = Eigen::Vector3d(0.0,0.0,0.0);

	// position of robots in world
	vector<Affine3d> robot_pose_in_world;
	Affine3d pose = Affine3d::Identity();
	pose.translation() = Vector3d(0.0, 0.0, 0.0);
	//pose.linear() = AngleAxisd(0, Vector3d::UnitY()).toRotationMatrix();
	robot_pose_in_world.push_back(pose);

	//------------------------------------------------------------
	// palm_posori_task
	Sai2Primitives::PosOriTask* palm_posori_task = new Sai2Primitives::PosOriTask(robot, link_name, pos_in_link);

	#ifndef USING_OTG
		palm_posori_task->_use_velocity_saturation_flag = true;
		palm_posori_task->_linear_saturation_velocity = 0.6;
		palm_posori_task->_angular_saturation_velocity = 45.0/180.0*M_PI;
	#endif

	Eigen::VectorXd posori_task_torques = Eigen::VectorXd::Zero(dof);
	palm_posori_task->_kp_pos = 500.0;
	palm_posori_task->_kv_pos = 20.0;
	palm_posori_task->_ki_pos = 2.0;
	palm_posori_task->_kp_ori = 500.0;
	palm_posori_task->_kv_ori = 20.0;
	palm_posori_task->_ki_ori = 2.0;

	Vector3d hand_desired_position_in_world; 
	Matrix3d hand_desired_orientation_in_world;

	//-------------------------------------------------------------
	//joint_posori_task
	Sai2Primitives::JointTask* joint_task = new Sai2Primitives::JointTask(robot);


	#ifdef USING_OTG
	bool _use_interpolation_flag=false;    // defaults to true
	// default limits for trajectory generation (same in all directions) :
	// Velocity      - PI/3  Rad/s
	// Acceleration  - PI    Rad/s^2
	// Jerk          - 3PI   Rad/s^3
	#endif

	joint_task->_kp = 100.0;
	joint_task->_kv = 30.0;
	Eigen::VectorXd task_joint_torques = Eigen::VectorXd::Zero(dof);



	string link_name2 = "link_3.0";
	Eigen::Vector3d pos_in_link2 = Eigen::Vector3d(0.0,0.0,0.0);

	//------------------------------------------------------------
	// finger_posori_task

	vector<Sai2Primitives::PosOriTask*> finger_posori_tasks;
	vector<VectorXd> finger_posori_task_torques;
	finger_posori_task_torques.push_back(Eigen::VectorXd::Zero(dof));
	finger_posori_task_torques.push_back(Eigen::VectorXd::Zero(dof));
	finger_posori_task_torques.push_back(Eigen::VectorXd::Zero(dof));

	vector<Vector3d> pos_in_links;
	pos_in_links.push_back(Eigen::Vector3d::Zero());
	pos_in_links.push_back(Eigen::Vector3d::Zero());
	pos_in_links.push_back(Eigen::Vector3d::Zero());

	vector<string> link_names;
	link_names.push_back("link_15.0");
	link_names.push_back("link_3.0");
	link_names.push_back("link_7.0");

	vector<Vector3d> finger_desired_position_in_world; 
	finger_desired_position_in_world.push_back(Eigen::Vector3d::Zero());
	finger_desired_position_in_world.push_back(Eigen::Vector3d::Zero());
	finger_desired_position_in_world.push_back(Eigen::Vector3d::Zero());

	vector<Matrix3d> finger_desired_orientation_in_world;
	finger_desired_orientation_in_world.push_back(Eigen::Matrix3d::Identity());
	finger_desired_orientation_in_world.push_back(Eigen::Matrix3d::Identity());
	finger_desired_orientation_in_world.push_back(Eigen::Matrix3d::Identity());
	for(int i=0 ; i<3 ; i++)
	{
		finger_posori_tasks.push_back(new Sai2Primitives::PosOriTask(robot, link_names[i], pos_in_link));
		finger_posori_task_torques.push_back(VectorXd::Zero(dof));

		finger_posori_tasks[i]->_kp_pos = 400.0;
		finger_posori_tasks[i]->_kv_pos = 25.0;
		finger_posori_tasks[i]->_kp_ori = 400.0;
		finger_posori_tasks[i]->_kv_ori = 40.0;		

	#ifndef USING_OTG
		finger_posori_tasks[i]->_use_velocity_saturation_flag = true;
		finger_posori_tasks[i]->_linear_saturation_velocity = 0.05;
		finger_posori_tasks[i]->_angular_saturation_velocity = 45.0/180.0*M_PI;
	#endif

	}


	//------------------------------------------------------------
	// Controller informations
	//------------------------------------------------------------
	double frequency = 1000;
	LoopTimer timer;
	timer.setLoopFrequency(frequency);
	timer.setCtrlCHandler(sighandler);
	//timer.initializeTimer(1000000);
	runloop = true ;
	int controller_counter = 0;
	int counter=0;

	while(runloop)
	{

		timer.waitForNextLoop();
		robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
		robot->_dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_KEY);
		robot->updateModel();
		robot->coriolisForce(coriolis);

		if (state == STREACH_HAND)
		{
			pregrasp_angles.setZero();
			joint_torques = compute_joint_cmd_torques(robot, pregrasp_angles);
			joint_torques.head(7).setZero();

    		if (joint_torques.norm() < 0.0001)
    		{
    			state = REACH_POSE;
    			cout << "hand stretched, lets go in position    " <<endl;
    			cout<<robot->_q<<endl;
    			robot->position(current_finger_position[0], link_name, pos_in_link);
    			cout<<current_finger_position[0]<<endl;
    		}	

		}


		if (state == REACH_POSE)
		{
			joint_task->reInitializeTask();
			joint_task->_kp = 300.0;
			joint_task->_kv = 30.0;
			joint_task->_desired_position.tail(16).setZero();		
			//pregrasp_angles.setZero();
			//joint_torques = compute_joint_cmd_torques(robot, pregrasp_angles);
			//joint_torques.head(7).setZero();


			//Update Nullspace matrixes
			N_prec.setIdentity();
			palm_posori_task->updateTaskModel(N_prec);
			N_prec = palm_posori_task->_N;
			joint_task->updateTaskModel(N_prec);
			//joint_task->_desired_position=pregrasp_angles;

			Eigen::Matrix3d R;
			R.setIdentity();
			//R=AngleAxisd(M_PI/4, Vector3d::UnitX()).toRotationMatrix();

			// set goal positions
			hand_desired_position_in_world = Vector3d(0.25, 0.25, 0.31);
			hand_desired_orientation_in_world = AngleAxisd(180.0/180.0*M_PI, Vector3d::UnitX()).toRotationMatrix()*AngleAxisd(0, Vector3d::UnitZ()).toRotationMatrix();
			
			palm_posori_task->_desired_position = robot_pose_in_world[0].linear().transpose()*(hand_desired_position_in_world - robot_pose_in_world[0].translation());
			palm_posori_task->_desired_orientation = robot_pose_in_world[0].linear().transpose()*hand_desired_orientation_in_world;		


			// torques
			palm_posori_task->computeTorques(posori_task_torques);
			joint_task->computeTorques(task_joint_torques);


    		if ((posori_task_torques+task_joint_torques).norm() + joint_torques.norm()< 0.005)
    		{
    			cout<<palm_posori_task->_desired_position<<endl;
    			robot->position(current_finger_position[0], link_name, pos_in_link);
    			cout<<current_finger_position[0]<<endl;
    			state = PRE_GRASP;

    			cout<<"we arrived, put the finger in a nicer position"<<endl<<endl;
    		}	
		}


		if (state == PRE_GRASP)
		{

			
			pregrasp_angles << 0,0,0,0,0,0,0, 0,0.9,1,0.85, 0,0.9,1,0.85, 0,0,0,0, -1.48,0.0,-0.28,1.4575;

			N_prec.setIdentity();
			palm_posori_task->updateTaskModel(N_prec);
			N_prec = palm_posori_task->_N;
			joint_task->updateTaskModel(N_prec);
			joint_task->_desired_position.tail(16)=pregrasp_angles.tail(16);
			palm_posori_task->computeTorques(posori_task_torques);
			joint_task->computeTorques(task_joint_torques);


    		if (posori_task_torques.norm() + task_joint_torques.norm() < 0.001)
    		{
    			state = APPROACH;
    			cout << "Close fingers" << endl;
    						// set goal positions

    			for(int i=0;i<3;i++){
				robot->rotation(finger_desired_orientation_in_world[i], link_names[i]);
				robot->position(finger_desired_position_in_world[i], link_names[i], pos_in_links[i]);
				cout << finger_desired_position_in_world[i]<< endl;
				}
				finger_desired_position_in_world[0][0]+=0.02;
				finger_desired_position_in_world[1][0]-=0.02;
				finger_desired_position_in_world[2][0]-=0.02;
				for( int i=0;i<3;i++){
				finger_posori_tasks[i]->_desired_position=finger_desired_position_in_world[i];
				finger_posori_tasks[i]->_desired_orientation=finger_desired_orientation_in_world[i];
				}
    		}	
		}

		if (state==APPROACH){

			N_prec.setIdentity();
			finger_posori_tasks[0]->updateTaskModel(N_prec);
			N_prec = finger_posori_tasks[0]->_N;
			finger_posori_tasks[1]->updateTaskModel(N_prec);
			//N_prec=finger_posori_tasks[1]->_N; //in _robot->operationalSpaceMatrices we have Ni = Ni - Jbar*task_jacobian; N = Ni*N_prec;
			finger_posori_tasks[2]->updateTaskModel(N_prec);
			N_prec=finger_posori_tasks[2]->_N;
			finger_posori_tasks[0]->computeTorques(finger_posori_task_torques[0]);
			finger_posori_tasks[1]->computeTorques(finger_posori_task_torques[1]);
			finger_posori_tasks[2]->computeTorques(finger_posori_task_torques[2]);
			posori_task_torques=finger_posori_task_torques[0]+finger_posori_task_torques[1]+finger_posori_task_torques[2];

			joint_task->reInitializeTask();
			joint_task->updateTaskModel(N_prec);
			joint_task->computeTorques(task_joint_torques);
			
			for(int i = 0; i < 3; i++)
			{
				if (finger_contact_flag[i] == 0)
				{
					Vector3d temp_finger_velocity = Vector3d::Zero();
					robot->linearVelocity(temp_finger_velocity, link_names[i], pos_in_link);
					velocity_record[i].pop_front();
					velocity_record[i].push_back(temp_finger_velocity.norm());
					if (velocity_record[i][1]/velocity_record[i][0] < 0.8 && velocity_record[i][0] > MIN_COLLISION_V)
					{
						cout <<"finger "<< i <<" contact"<<endl;
						cout << "the previous velocity is: " << velocity_record[i][0] << endl;
						cout << "the current velocity is: " << velocity_record[i][1] << endl;
						finger_contact_flag[i] = 1;
						// set the desired position, maintain the current position
						robot->rotation(finger_desired_orientation_in_world[i], link_names[i]);
						robot->position(finger_desired_position_in_world[i], link_names[i], pos_in_links[i]);
						if (i==0){finger_desired_position_in_world[i][0]+=0.002;}
						else finger_desired_position_in_world[i][0]-=0.002;
						
						finger_posori_tasks[i]->_desired_position=finger_desired_position_in_world[i];
						finger_posori_tasks[i]->_desired_orientation=finger_desired_orientation_in_world[i];
						// cout << current_finger_position[i] << endl;
					}
				}
			}

			cout<<robot->_dq.norm()<<endl;
   			//if (finger_contact_flag[0] == 1 && finger_contact_flag[1] == 1 && finger_contact_flag[2] == 1)
   			//{
   			//robot->position(finger_desired_position_in_world[0], link_names[0], pos_in_links[0]);
   			//	if((finger_desired_position_in_world[0]-finger_posori_tasks[0]->_desired_position).norm()<0.0001){
   			//		state = CENTER_THUMB;
   			//		cout<<"roll the thumb"<<endl<<endl;
   			//	}
   			//}	
   			if (finger_contact_flag[0] == 1 && finger_contact_flag[1] == 1 && finger_contact_flag[2] == 1)
   			{
   				if(robot->_dq.norm()<0.03){
   					state = CENTER_THUMB;
   					cout<<"roll the thumb"<<endl<<endl;
   				}
   			}	
		}

		if (state==CENTER_THUMB){


			pos_in_link=Vector3d(0,0,0.0423)+Vector3d(1,0,1)/Vector3d(1,0,1).norm()*0.012;
			finger_posori_tasks[0]=new Sai2Primitives::PosOriTask(robot, link_names[0], pos_in_link);
			Vector3d tip; 
			robot->position(tip,link_names[0],pos_in_link);
			robot->position(finger_desired_position_in_world[0], link_names[0], Vector3d(0,0,0.0423));
			finger_posori_tasks[0]->_desired_position=tip+(tip-finger_desired_position_in_world[0])/(tip-finger_desired_position_in_world[0]).norm()*0.01;

			//robot->rotation(finger_desired_orientation_in_world[0], link_names[0]);
			//finger_posori_tasks[0]->_desired_orientation=finger_desired_orientation_in_world[0];

			N_prec.setIdentity();
			finger_posori_tasks[0]->updateTaskModel(N_prec);
			N_prec = finger_posori_tasks[0]->_N;			
			finger_posori_tasks[1]->updateTaskModel(N_prec);
			//N_prec = finger_posori_tasks[2]->_N;
			//N_prec=finger_posori_tasks[1]->_N; //in _robot->operationalSpaceMatrices we have Ni = Ni - Jbar*task_jacobian; N = Ni*N_prec;
			finger_posori_tasks[2]->updateTaskModel(N_prec);
			N_prec=finger_posori_tasks[2]->_N;
			N_prec*=finger_posori_tasks[1]->_N;

			finger_posori_tasks[0]->_kp_pos = 1000.0;
			finger_posori_tasks[0]->_kv_pos = 25.0;
			finger_posori_tasks[0]->_kp_ori = 20.0;
			finger_posori_tasks[0]->_kv_ori = 40.0;		
			finger_posori_tasks[1]->_kp_pos = 500.0;
			finger_posori_tasks[1]->_kv_pos = 25.0;
			finger_posori_tasks[1]->_kp_ori = 500.0;
			finger_posori_tasks[1]->_kv_ori = 40.0;	
			finger_posori_tasks[2]->_kp_pos = 500.0;
			finger_posori_tasks[2]->_kv_pos = 25.0;
			finger_posori_tasks[2]->_kp_ori = 500.0;
			finger_posori_tasks[2]->_kv_ori = 40.0;	

			finger_posori_tasks[0]->computeTorques(finger_posori_task_torques[0]);
			finger_posori_tasks[1]->computeTorques(finger_posori_task_torques[1]);
			finger_posori_tasks[2]->computeTorques(finger_posori_task_torques[2]);
			posori_task_torques=finger_posori_task_torques[0]+finger_posori_task_torques[1]+finger_posori_task_torques[2];

			joint_task->reInitializeTask();
			joint_task->updateTaskModel(N_prec);
			joint_task->computeTorques(task_joint_torques);


		}

		if (state==LIFT){

			joint_task->_kp = 100.0;
			joint_task->_kv = 30.0;
			
			N_prec.setIdentity();
			palm_posori_task->updateTaskModel(N_prec);
			N_prec = palm_posori_task->_N;
			joint_task->updateTaskModel(N_prec);
			joint_task->_desired_position.head(7)=robot->_q.head(7);
			palm_posori_task->_linear_saturation_velocity = 0.01;
			palm_posori_task->_angular_saturation_velocity = 5.0/180.0*M_PI;

			Eigen::Matrix3d R;
			R.setIdentity();
			//R=AngleAxisd(M_PI/4, Vector3d::UnitX()).toRotationMatrix();

			// set goal positions
			hand_desired_position_in_world = Vector3d(0.25, 0.25, 0.45);
			hand_desired_orientation_in_world = AngleAxisd(180.0/180.0*M_PI, Vector3d::UnitX()).toRotationMatrix()*AngleAxisd(0, Vector3d::UnitZ()).toRotationMatrix();
			
			palm_posori_task->_desired_position = robot_pose_in_world[0].linear().transpose()*(hand_desired_position_in_world - robot_pose_in_world[0].translation());
			palm_posori_task->_desired_orientation = robot_pose_in_world[0].linear().transpose()*hand_desired_orientation_in_world;		


			// torques
			palm_posori_task->computeTorques(posori_task_torques);
			joint_task->computeTorques(task_joint_torques);



    		if (robot->_dq.norm() < 0.001)
    		{
    			state = LIFT;
    			cout<<"we arrived, put the finger in a nicer position"<<endl<<endl;
    		}	
		}




		//command_torques=finger_command_torques;
		robot->gravityVector(gravity_torques);
		//command_torques = finger_command_torques[0] + finger_command_torques[1] \
		+ finger_command_torques[2] + finger_command_torques[3]+ gravity_torques;

		//command_torques =joint_torques;
		//robot->gravityVector(command_torques);
		command_torques = posori_task_torques + task_joint_torques  + finger_command_torques[0] + finger_command_torques[1] \
		+ finger_command_torques[2] + finger_command_torques[3]+joint_torques+coriolis;

		redis_client.setEigenMatrixJSON(JOINT_TORQUES_COMMANDED_KEY, command_torques);

		controller_counter++;

		JacobiSVD<MatrixXd> svd(robot->_M);
		double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);

		if(controller_counter % 4000 == 0)
		{
			cout <<  " finger 0" <<endl << finger_posori_task_torques[0]<< endl<<endl;
			cout << "finger 1" <<endl<<finger_posori_task_torques[1] << endl << endl;
			cout << "finger2" <<finger_posori_task_torques[2]<< endl << endl;
			cout << "task joint torques: " <<endl<< task_joint_torques<< endl<<endl;
			//cout << joint_task->_current_position - joint_task->_step_desired_position << endl <<endl;

			cout << endl;
		}


		// reset to 0
		
		for(int i =0; i < NUM_OF_FINGERS_IN_MODEL; i++)
		{
			temp_finger_command_torques[i].setZero();
			finger_command_torques[i].setZero();
		}
		posori_task_torques.setZero();
		command_torques.setZero();
		joint_torques.setZero();
		task_joint_torques.setZero();
		posori_task_torques.setZero();
	}

	command_torques.setZero();
    //redis_client.setEigenMatrixDerived(JOINT_TORQUES_COMMANDED_KEY, command_torques);

    double end_time = timer.elapsedTime();
    std::cout << "\n";
    std::cout << "Loop run time  : " << end_time << " seconds\n";
    std::cout << "Loop updates   : " << timer.elapsedCycles() << "\n";
    std::cout << "Loop frequency : " << timer.elapsedCycles()/end_time << "Hz\n";

	return 0;
}

VectorXd compute_position_cmd_torques(Sai2Model::Sai2Model* robot, string link, Vector3d pos_in_link, Vector3d desired_position, double kp)
{
	// double kp = 10;
	double kv = kp/5;
	double damping = -0.005;

	int dof = robot->dof();
	Vector3d current_position; // in robot frame
	Vector3d current_velocity;
	MatrixXd Jv = MatrixXd::Zero(3,dof);
	robot->Jv(Jv, link, pos_in_link);
	robot->position(current_position, link, pos_in_link);
	robot->linearVelocity(current_velocity, link, pos_in_link);
	VectorXd torque = VectorXd::Zero(dof);
	torque = Jv.transpose()*(kp*(desired_position - current_position) - kv * current_velocity) + damping * robot->_dq;
	return torque;
}

VectorXd compute_force_cmd_torques(Sai2Model::Sai2Model* robot, string link, Vector3d pos_in_link, Vector3d desired_position, double force_requeired)
{
	int dof = robot->dof();
	// double force_requeired = 0.001;
	double damping = -force_requeired / 10;
	Vector3d current_position; // in robot frame
	Vector3d current_velocity;
	Vector3d desired_force;
	MatrixXd Jv = MatrixXd::Zero(3,dof);
	robot->Jv(Jv, link, pos_in_link);
	robot->position(current_position, link, pos_in_link);
	desired_force = desired_position - current_position;
	desired_force = desired_force / desired_force.norm(); // normalization
	desired_force = desired_force * force_requeired;
	VectorXd torque = VectorXd::Zero(dof);
	torque = Jv.transpose()*desired_force + damping * robot->_dq;
	return torque;
}



bool check_2_finger_grasp(vector<Vector3d> contact_points,vector<Vector3d> normals, double friction_coefficient)
{
	double alpha;
	alpha = atan(friction_coefficient);
	Vector3d connect_vector = Vector3d::Zero();
	contact_points.push_back(contact_points[0]);
	normals.push_back(normals[0]);
	int flag = 0;
	for(int i = 0; i < NUM_OF_FINGERS_USED; i++)
	{
		flag = 0;
		connect_vector = contact_points[i+1] - contact_points[i];
		if(normals[i].dot(connect_vector)/(normals[i].norm() * connect_vector.norm()) > cos(alpha));
		{
			flag++;
		}
		if(-normals[i+1].dot(connect_vector)/(normals[i+1].norm() * connect_vector.norm()) < cos(alpha))
		{
			flag++;
		}
		if (flag == 2)
		{
			return true;
		}
	}
	return false;
}

bool check_3_finger_grasp(vector<Vector3d> contact_points,vector<Vector3d> normals, double friction_coefficient)
{
	// first, we need to transform the problem into a palnar problem
	//the first step is to use Schmidt orthogonalization to get a standard unit base
	Vector3d base_1 = contact_points[1] - contact_points[0];
	Vector3d temp = contact_points[2] - contact_points[0];
	Vector3d base_2 = temp - temp.dot(base_1) / base_1.dot(base_1) * base_1;
	Vector3d base_3 = base_1.cross(base_2);
	base_1 = base_1 / base_1.norm(); // normalization
	base_2 = base_2 / base_2.norm();
	base_3 = base_3 / base_3.norm();
	// get the transformation matrix
	Matrix3d R_inverse = Matrix3d::Zero();
	R_inverse.block(0,0,1,3) = base_1;
	R_inverse.block(0,1,1,3) = base_2;
	R_inverse.block(0,2,1,3) = base_3;
	Matrix3d R = R_inverse.inverse(); // R is the trasformation matrix from the original base to the current base
	vector<double> thetas;
	for ( int i = 0; i < NUM_OF_FINGERS_USED; i++)
	{
		contact_points[i] = R * contact_points[i];
		normals[i] = R * normals[i];
		thetas.push_back(asin(normals[i][2]));
	}
	

	return true;
}
bool check_positive_span_space(vector<Vector3d> normals)
{
	for (int i = 0; i < NUM_OF_FINGERS_USED; i++)
	{
		normals.push_back(normals[i]);
	}
	int flag = 0;
	for (int i = 0; i < NUM_OF_FINGERS_USED; i++)
	{
		if(normals[i].cross(normals[i+1]).dot(normals[i].cross(normals[i+2])) < 0)
		{
			flag++;
		}

	}
	if(flag == 3)
	{
		return true;
	}
	else
	{
		return false;
	}
}

VectorXd compute_joint_cmd_torques(Sai2Model::Sai2Model* robot, VectorXd desired_joint_angles)
{
  double kp = 0.1;
  double kv = 0.005;
  VectorXd q = robot->_q;
  VectorXd dq = robot->_dq;
  int dof = robot->dof();
  VectorXd torques = VectorXd::Zero(dof);
  for(int i = 0; i < dof; i++)
  {
    torques[i] = kp * (desired_joint_angles[i] - q[i]) - kv * dq[i];
  }
  return torques;
}
VectorXd compute_joint_cmd_torques(Sai2Model::Sai2Model* robot, VectorXd desired_joint_angles, double kp)
{
  
  double kv = kp/5;
  VectorXd q = robot->_q;
  VectorXd dq = robot->_dq;
  int dof = robot->dof();
  VectorXd torques = VectorXd::Zero(dof);
  for(int i = 0; i < dof; i++)
  {
    torques[i] = kp * (desired_joint_angles[i] - q[i]) - kv * dq[i];
  }
  return torques;
}
VectorXd compute_joint_cmd_torques_one_finger(Sai2Model::Sai2Model* robot, VectorXd desired_joint_angles, int index)
{
  int dof = robot->dof();
  VectorXd torques = compute_joint_cmd_torques(robot, desired_joint_angles);
  for( int i = 0; i < dof; i++)
  {
    if((i < index * 4 + 10) && (i >= index * 4 + 6))
    {
      continue;
    }
    else
    {
      torques[i] = 0;
    }
  }
  return torques;
}