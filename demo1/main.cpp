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

const string robot_file = "./resources/hand.urdf";
const string robot_name = "Hand3Finger";


const std::string JOINT_ANGLES_KEY  = "sai2::graspM::sensors::q";
const std::string JOINT_VELOCITIES_KEY = "sai2::graspM::sensors::dq";
const std::string JOINT_TORQUES_COMMANDED_KEY = "sai2::graspM::actuators::fgc";

#define NUM_OF_FINGERS_IN_MODEL 4
#define NUM_OF_FINGERS_USED     3

#define CONTACT_COEFFICIENT     0.5 
#define MIN_COLLISION_V         0.01
#define DISPLACEMENT_DIS        0.02  // how much you wanna move awat from the original point in normal detection step
#define FRICTION_COEFFICIENT     0.5

#define PRE_GRASP               0
#define FINGER_MOVE_CLOSE       1
#define DETECT_NORMAL           2
#define CHECK                   3
#define APPLY_FORCE             4
#define LIFT                    5

int state = PRE_GRASP;

double prob_distance = 0.006; // how much you want the to prob laterally in normal detection step

// the function used in the finger position control command
VectorXd compute_position_cmd_torques(Sai2Model::Sai2Model* robot, string link, Vector3d pos_in_link, Vector3d desired_position, double kp);
// the function used in the finger force control, used to achieve compliance
VectorXd compute_force_cmd_torques(Sai2Model::Sai2Model* robot, string link, Vector3d pos_in_link, Vector3d desired_position, double force_requeired);
// this function is used to detect surface normal by sampling several points in the vicinity
// It can only be called when the finger tip is making contact with the object surface
// returns the torque needed 
VectorXd detect_surface_normal(Sai2Model::Sai2Model* robot, string link, Vector3d pos_in_link, Vector3d original_pos, Vector3d CoM_of_object, int& state, deque<double>& velocity_record, vector<Vector3d>& contact_points, Vector3d& normal);
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

	double frequency = 1000;

	auto robot = new Sai2Model::Sai2Model(robot_file, false);

	// read from Redis
	robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
	robot->_dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_KEY);
	robot->updateModel();
	int dof = robot->dof();

	vector<Vector3d> current_finger_position;
	VectorXd command_torques = VectorXd::Zero(dof);
	vector<VectorXd> finger_command_torques;
	VectorXd palm_command_torques = VectorXd::Zero(dof);
	VectorXd coriolis = VectorXd::Zero(dof);
	MatrixXd N_prec = MatrixXd::Identity(dof,dof);
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
	vector<string> link_names;
	vector<Affine3d> poses;
	Affine3d identity_pose = Affine3d::Identity();
	Affine3d temp_pose = Affine3d::Identity();
	temp_pose.translation() = Vector3d(0.0507,0.0,0.0);
	poses.push_back(temp_pose);
	temp_pose.translation() = Vector3d(0.0327, 0.0, 0.0);
	poses.push_back(temp_pose);
	poses.push_back(temp_pose);
	poses.push_back(temp_pose);

	link_names.push_back("finger0-link3");
	link_names.push_back("finger1-link3");
	link_names.push_back("finger2-link3");
	link_names.push_back("finger3-link3");

	Vector3d CoM_of_object = Vector3d(0.05,0.0,0.05); // in the world frame
	CoM_of_object -= Vector3d(0.0, 0.0, 0.25); // transform into the robor frame

	for(int i = 0; i < NUM_OF_FINGERS_USED; i++)
	{
		deque<double> temp_queue;
		temp_queue.push_back(0.0);
		temp_queue.push_back(0.0);
		velocity_record.push_back(temp_queue);
		detect_velocity_record.push_back(temp_queue);

		current_finger_position.push_back(Vector3d::Zero());

		detect_states.push_back(0);

		normals.push_back(Vector3d::Zero());
	}

	auto palm_posori_task = new Sai2Primitives::PosOriTask(robot, "palm", Vector3d(0.0,0.0,0.0));

	LoopTimer timer;
	timer.setLoopFrequency(frequency);
	timer.setCtrlCHandler(sighandler);
	//timer.initializeTimer(1000000);

	vector<int> finger_contact_flag; // finger0, 1, 2, 3
	for (int i = 0; i < NUM_OF_FINGERS_IN_MODEL; i++)
	{
		finger_contact_flag.push_back(0);
	}

	runloop = true ;
	int loop_counter = 0;


	// cout << robot->_joint_names_map["finger0-j0"] << "!!!!!!!!!!!!!!!!!!!!!!!" <<endl;
	while(runloop)
	{

		timer.waitForNextLoop();
		robot->_q = redis_client.getEigenMatrixJSON(JOINT_ANGLES_KEY);
		robot->_dq = redis_client.getEigenMatrixJSON(JOINT_VELOCITIES_KEY);
		//cout <<"q" << robot->_q << endl;
		robot->updateModel();
		robot->coriolisForce(coriolis);

		if (state == PRE_GRASP)
		{
			palm_posori_task->_desired_position = Vector3d(0.03,0.0,-0.08);
			N_prec.setIdentity();
			palm_posori_task->updateTaskModel(N_prec);
			N_prec = palm_posori_task->_N;
			palm_posori_task->computeTorques(palm_command_torques);
			//cout << "Here's the torque" << palm_command_torques << endl;
			temp_finger_command_torques[0] = compute_position_cmd_torques(robot, link_names[0], poses[0].translation(), Vector3d(-0.08, 0.0, -0.15), 10.0);
			temp_finger_command_torques[1] = compute_position_cmd_torques(robot, link_names[1], poses[1].translation(), Vector3d(0.15, -0.041, -0.2), 10.0);
    		temp_finger_command_torques[2] = compute_position_cmd_torques(robot, link_names[2], poses[2].translation(), Vector3d(0.15, 0.0, -0.2), 10.0);
    		temp_finger_command_torques[3] = compute_position_cmd_torques(robot, link_names[3], poses[3].translation(), Vector3d(0.15, 0.041, -0.09), 10.0);
		    		
    		// block the unrelated torques
    		finger_command_torques[0].block(6,0,4,1) = temp_finger_command_torques[0].block(6,0,4,1);
    		finger_command_torques[1].block(10,0,4,1) = temp_finger_command_torques[1].block(10,0,4,1);
    		finger_command_torques[2].block(14,0,4,1) = temp_finger_command_torques[2].block(14,0,4,1);
    		finger_command_torques[3].block(18,0,4,1) = temp_finger_command_torques[3].block(18,0,4,1);


    		if (palm_command_torques.norm() + finger_command_torques[0].norm() + finger_command_torques[1].norm() + finger_command_torques[2].norm() < 0.0001)
    		{
    			state = FINGER_MOVE_CLOSE;
    		}
		}
		else if (state == FINGER_MOVE_CLOSE)
		{	
			// keep the position of the palm
			palm_posori_task->_desired_position = Vector3d(0.03,0.0,-0.08);
			N_prec.setIdentity();
			palm_posori_task->updateTaskModel(N_prec);
			N_prec = palm_posori_task->_N;
			palm_posori_task->computeTorques(palm_command_torques);

			// force controller for the fingers
			for(int i = 0; i < NUM_OF_FINGERS_USED; i++)
			{
				if (finger_contact_flag[i] == 0)
				{
					temp_finger_command_torques[i] = compute_force_cmd_torques(robot, link_names[i], poses[i].translation(), CoM_of_object, 0.001);
					finger_command_torques[i].block(6+4*i,0,4,1) = temp_finger_command_torques[i].block(6+4*i,0,4,1);
					Vector3d temp_finger_velocity = Vector3d::Zero();
					robot->linearVelocity(temp_finger_velocity, link_names[i], poses[i].translation());
					velocity_record[i].pop_front();
					velocity_record[i].push_back(temp_finger_velocity.norm());
					if (velocity_record[i][1]/velocity_record[i][0] < 0.5 && velocity_record[i][0] > MIN_COLLISION_V)
					{
						cout <<"finger "<< i <<" contact"<<endl;
						cout << "the previous velocity is: " << velocity_record[i][0] << endl;
						cout << "the current velocity is: " << velocity_record[i][1] << endl;
						finger_contact_flag[i] = 1;
						// set the desired position, maintain the current position
						robot->position(current_finger_position[i], link_names[i], poses[i].translation());
						// cout << current_finger_position[i] << endl;
					}
				}
				// maintain the current position after contact
				else if (finger_contact_flag[i] == 1)
				{
					temp_finger_command_torques[i] = compute_position_cmd_torques(robot, link_names[i], poses[i].translation(), current_finger_position[i], 10.0);
    				finger_command_torques[i].block(6 + 4 * i ,0 ,4, 1) = temp_finger_command_torques[i].block(6 + 4 * i, 0 ,4 ,1 );
				} 
			}

			// keep the position of fingers that are not used
			for (int j = NUM_OF_FINGERS_USED; j < NUM_OF_FINGERS_IN_MODEL; j++)
			{
	    		temp_finger_command_torques[j] = compute_position_cmd_torques(robot, link_names[j], poses[j].translation(), Vector3d(0.15, 0.041, -0.09), 10.0);
				finger_command_torques[j].block(6+4*j,0,4,1) = temp_finger_command_torques[j].block(6+4*j,0,4,1);
			}

			int sum_of_contact = 0;
			for (int j = 0; j < NUM_OF_FINGERS_USED; j++)
			{
				sum_of_contact += finger_contact_flag[j];
			}
			if (sum_of_contact == NUM_OF_FINGERS_USED)
			{
				state = DETECT_NORMAL;
				for (int j = 0; j < NUM_OF_FINGERS_USED; j++)
				{
					contact_points.push_back({current_finger_position[j]});
				}
			}
		}

		else if (state == DETECT_NORMAL)
		{
			double sum_of_normal = 0.0;
			palm_posori_task->_desired_position = Vector3d(0.03,0.0,-0.08);
			N_prec.setIdentity();
			palm_posori_task->updateTaskModel(N_prec);
			N_prec = palm_posori_task->_N;
			palm_posori_task->computeTorques(palm_command_torques);
			for (int i = 0; i < NUM_OF_FINGERS_USED; i++)
			{
				temp_finger_command_torques[i] = detect_surface_normal(robot, link_names[i], poses[i].translation(), current_finger_position[i], CoM_of_object, detect_states[i], detect_velocity_record[i], contact_points[i], normals[i]);
 				finger_command_torques[i].block(6 + 4 * i ,0 ,4, 1) = temp_finger_command_torques[i].block(6 + 4 * i, 0 ,4 ,1 );
				sum_of_normal += normals[i].norm();
/*				cout << normals[i] << endl;
				if (i == 0)
				{
					cout << finger_command_torques[i].block(6 + 4 * i ,0 ,4, 1) << endl;
				}*/
			}
/*			cout << endl << endl;
			if(loop_counter % 1000 == 0)
			{
				for(int j = 0; j < normals.size(); j++)
				{
					cout <<"normals" << j << ":\n" << normals[j] << endl;
				}
				cout << sum_of_normal << "!!!!!!" << endl;
			}*/
			for (int j = NUM_OF_FINGERS_USED; j < NUM_OF_FINGERS_IN_MODEL; j++)
			{
	    		temp_finger_command_torques[j] = compute_position_cmd_torques(robot, link_names[j], poses[j].translation(), Vector3d(0.15, 0.041, -0.09), 10.0);
				finger_command_torques[j].block(6+4*j,0,4,1) = temp_finger_command_torques[j].block(6 + 4 * j,0,4,1);
			}
			//cout << sum_of_normal << endl;
			if(sum_of_normal > double(NUM_OF_FINGERS_USED)-0.5)
			{
				cout << "all the normals detected" << endl;
				state = CHECK;
			}

		}

		else if (state == CHECK)
		{
			// check whether we can achieve 2 finger contact.
			if (check_2_finger_grasp(current_finger_position, normals, FRICTION_COEFFICIENT))
				{state = APPLY_FORCE;}


		}

		else if (state == APPLY_FORCE)
		{
			palm_posori_task->_desired_position = Vector3d(0.03,0.0,-0.08);
			palm_posori_task->_kp_force = 500.0;
			palm_posori_task->_kp_moment = 500.0;
			palm_posori_task->_kv_force = 10.0;
			palm_posori_task->_kv_moment = 10.0;
			palm_posori_task->_ki_force = 10.0;
			palm_posori_task->_ki_moment = 10.0;
			N_prec.setIdentity();
			palm_posori_task->updateTaskModel(N_prec);
			N_prec = palm_posori_task->_N;
			palm_posori_task->computeTorques(palm_command_torques);
			for(int j = 1; j < NUM_OF_FINGERS_USED; j++)
			{
				temp_finger_command_torques[j] = compute_position_cmd_torques(robot, link_names[j], poses[j].translation(), current_finger_position[j], 100.0);
			}
			robot->position(current_finger_position[0], link_names[0], poses[0].translation());
			temp_finger_command_torques[0] = compute_force_cmd_torques(robot, link_names[0], poses[0].translation(), current_finger_position[0] + normals[0], 0.3);
			for(int j = 0; j < NUM_OF_FINGERS_USED; j++)
			{
    			finger_command_torques[j].block(6 + 4 * j ,0 ,4, 1) = temp_finger_command_torques[j].block(6 + 4 * j, 0 ,4 ,1 );
			}

			for (int j = NUM_OF_FINGERS_USED; j < NUM_OF_FINGERS_IN_MODEL; j++)
			{
	    		temp_finger_command_torques[j] = compute_position_cmd_torques(robot, link_names[j], poses[j].translation(), Vector3d(0.15, 0.041, -0.09), 10.0);
				finger_command_torques[j].block(6+4*j,0,4,1) = temp_finger_command_torques[j].block(6+4*j,0,4,1);
			}
		}

		loop_counter++;

	command_torques = finger_command_torques[0] + finger_command_torques[1] \
	+ finger_command_torques[2] + finger_command_torques[3]\
	+ palm_command_torques + coriolis;
	//cout << command_torques << endl;
	redis_client.setEigenMatrixJSON(JOINT_TORQUES_COMMANDED_KEY, command_torques);
		// reset to 0
	for(int i =0; i < NUM_OF_FINGERS_IN_MODEL; i++)
	{
		temp_finger_command_torques[i].setZero();
		finger_command_torques[i].setZero();
	}
	command_torques.setZero();
	}

	command_torques.setZero();
    redis_client.setEigenMatrixDerived(JOINT_TORQUES_COMMANDED_KEY, command_torques);

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

VectorXd detect_surface_normal(Sai2Model::Sai2Model* robot, string link, Vector3d pos_in_link, Vector3d original_pos, Vector3d CoM_of_object, int& state, deque<double>& velocity_record, vector<Vector3d>& contact_points, Vector3d& normal)
{
	int dof = robot->dof();
	VectorXd torque = VectorXd::Zero(dof);
	Vector3d current_position = Vector3d::Zero();
	robot->position(current_position, link, pos_in_link);
	if(state == 0) // just start from the initial centroid position
	{

		Vector3d desired_position = DISPLACEMENT_DIS*(original_pos - CoM_of_object) / (original_pos - CoM_of_object).norm() + \
		original_pos + Vector3d(0.0, 0.0, prob_distance);
		torque = compute_position_cmd_torques(robot, link, pos_in_link, desired_position, 10.0);
		if((desired_position - current_position).norm() < 0.001)
		{
			state = 1;
		}
	}
	else if (state == 1) // has reached the first intermediate point
	{
		torque = compute_force_cmd_torques(robot, link, pos_in_link, CoM_of_object, 0.001);
		Vector3d temp_finger_velocity = Vector3d::Zero();
		robot->linearVelocity(temp_finger_velocity, link, pos_in_link);
		velocity_record.pop_front();
		velocity_record.push_back(temp_finger_velocity.norm());
		if (velocity_record[1]/velocity_record[0] < CONTACT_COEFFICIENT && velocity_record[0] > MIN_COLLISION_V)
		{
			state = 2;
			cout << link <<" contact"<<endl;
			cout<< "the previous velocity is: " << velocity_record[0] << endl;
			cout << "the current velocity is: " << velocity_record[1] << endl;
			contact_points.push_back(current_position);
		}
	}

	else if(state == 2) 
	{

		Vector3d desired_position = DISPLACEMENT_DIS*(original_pos - CoM_of_object) / (original_pos - CoM_of_object).norm() + \
		original_pos + Vector3d(0.0, 0.0, -prob_distance);
		torque = compute_position_cmd_torques(robot, link, pos_in_link, desired_position, 10.0);
		if((desired_position - current_position).norm() < 0.001)
		{
			state = 3;
		}
	}

	else if (state == 3) // has reached the second intermediate point
	{
		torque = compute_force_cmd_torques(robot, link, pos_in_link, CoM_of_object, 0.001);
		Vector3d temp_finger_velocity = Vector3d::Zero();
		robot->linearVelocity(temp_finger_velocity, link, pos_in_link);
		velocity_record.pop_front();
		velocity_record.push_back(temp_finger_velocity.norm());
		if (velocity_record[1]/velocity_record[0] < CONTACT_COEFFICIENT && velocity_record[0] > MIN_COLLISION_V)
		{
			state = 4;
			cout << link <<" contact"<<endl;
			cout<< "the previous velocity is: " << velocity_record[0] << endl;
			cout << "the current velocity is: " << velocity_record[1] << endl;
			contact_points.push_back(current_position);	
			// cout << contact_points[100] << "test" << endl;		
		}
	}

	else if(state == 4) 
	{
		Vector3d disp = Vector3d(0.0, 0.0, 0.0);
		Vector3d origin_disp = (original_pos - CoM_of_object) / (original_pos - CoM_of_object).norm();
		disp[0] = origin_disp[1]/sqrt(pow(origin_disp[0], 2) + pow(origin_disp[1], 2));
		disp[1] = - origin_disp[0]/sqrt(pow(origin_disp[0], 2) + pow(origin_disp[1], 2));

		Vector3d desired_position = DISPLACEMENT_DIS*(original_pos - CoM_of_object) / (original_pos - CoM_of_object).norm() + \
		original_pos + prob_distance * disp;
		torque = compute_position_cmd_torques(robot, link, pos_in_link, desired_position, 10.0);
		if((desired_position - current_position).norm() < 0.001)
		{
			state = 5;
		}
	}

	else if (state == 5) // has reached the second intermediate point
	{
		torque = compute_force_cmd_torques(robot, link, pos_in_link, CoM_of_object, 0.001);
		Vector3d temp_finger_velocity = Vector3d::Zero();
		robot->linearVelocity(temp_finger_velocity, link, pos_in_link);
		velocity_record.pop_front();
		velocity_record.push_back(temp_finger_velocity.norm());
		if (velocity_record[1]/velocity_record[0] < CONTACT_COEFFICIENT && velocity_record[0] > MIN_COLLISION_V)
		{
			state = 6;
			cout << link <<" contact"<<endl;
			cout<< "the previous velocity is: " << velocity_record[0] << endl;
			cout << "the current velocity is: " << velocity_record[1] << endl;

			contact_points.push_back(current_position);	
			// cout << contact_points[100] << "test" << endl;		
		}
	}

	else if(state == 6) 
	{
		Vector3d disp = Vector3d(0.0, 0.0, 0.0);
		Vector3d origin_disp = (original_pos - CoM_of_object) / (original_pos - CoM_of_object).norm();
		disp[0] = origin_disp[1]/sqrt(pow(origin_disp[0], 2) + pow(origin_disp[1], 2));
		disp[1] = - origin_disp[0]/sqrt(pow(origin_disp[0], 2) + pow(origin_disp[1], 2));
		disp = - disp;

		Vector3d desired_position = DISPLACEMENT_DIS*(original_pos - CoM_of_object) / (original_pos - CoM_of_object).norm() + \
		original_pos + prob_distance * disp;
		torque = compute_position_cmd_torques(robot, link, pos_in_link, desired_position, 10.0);
		if((desired_position - current_position).norm() < 0.001)
		{
			state = 7;
		}
	}

	else if (state == 7) // has reached the fourth intermediate point
	{
		torque = compute_force_cmd_torques(robot, link, pos_in_link, CoM_of_object, 0.001);
		Vector3d temp_finger_velocity = Vector3d::Zero();
		robot->linearVelocity(temp_finger_velocity, link, pos_in_link);
		velocity_record.pop_front();
		velocity_record.push_back(temp_finger_velocity.norm());
		if (velocity_record[1]/velocity_record[0] < CONTACT_COEFFICIENT && velocity_record[0] > MIN_COLLISION_V)
		{
			state = 8;
			cout << link <<" contact "<<endl;
			cout<< "the previous velocity is: " << velocity_record[0] << endl;
			cout << "the current velocity is: " << velocity_record[1] << endl;

			contact_points.push_back(current_position);	
			// cout << contact_points[100] << "test" << endl;		
		}
	}

	else if (state == 8) // go back to the original contact position
	{
		torque = compute_position_cmd_torques(robot, link, pos_in_link, original_pos, 10.0);
		if((original_pos - current_position).norm() < 0.01)
		{
			state = 9;
		}
	}

	else if (state == 9)  // compute the normal
	{
		cout << "I am computing the normal for "<< link << endl; 
		// cout << "contact_points" << endl;
		Matrix3d coefficient_matrix = Matrix3d::Zero();
		Vector3d mean_position = Vector3d(0.0, 0.0, 0.0);
		auto centralized_position = contact_points;
		for (int j = 0; j < contact_points.size(); j++ )
		{
			mean_position += contact_points[j];
			// cout << contact_points[j] <<endl<< endl;
		}
		mean_position /= contact_points.size();
		for (int j = 0; j < contact_points.size(); j++)
		{
			centralized_position[j] -= mean_position;
		}
		for (int j = 0; j < contact_points.size(); j++)
		{
			coefficient_matrix += centralized_position[j] * centralized_position[j].transpose();
		}
		EigenSolver<Matrix3d> solver(coefficient_matrix);
		Matrix3d eigen_matrix = solver.eigenvectors().real();
		Vector3d eigen_values = solver.eigenvalues().real();
		int min_index = 999;
		double min_value = 999;
		for (int j = 0; j < 3; j++)
		{
			if (eigen_values[j] < min_value)
			{
				min_value = eigen_values[j];
				min_index = j;
			}
		}
		normal = eigen_matrix.real().col(min_index);
		// the following code chose which direction should the normal choose
		// it's the direction position to the CoM
		Vector3d temp = CoM_of_object - original_pos;
		// cout << normal << endl << endl;
		if ( temp.dot(normal) < 0)
		{
			normal = -normal; // opposite position
		}
		cout << "Here is the normal for finger " << link << endl << normal << endl << endl;
		state = 10;
		// cout << "the normal is "<< link << endl << normal << endl << endl;
		// cout << "the eigen vectors are" << endl << eigen_matrix << endl;
		// cout << "the eigen values are " << endl << eigen_values << endl;
	}
	else if (state == 10) // maintain the original contact position
	{
		torque = compute_position_cmd_torques(robot, link, pos_in_link, original_pos, 10.0);
	}
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