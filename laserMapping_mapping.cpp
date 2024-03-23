#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <ros/ros.h>
#include <Eigen/Core>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>

#include "IMU_Processing.hpp"

// gstam
#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <pcl/search/impl/search.hpp>
#include <pcl/range_image/range_image.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/crop_box.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include "GNSS_Processing.hpp"

#include "na_mapping/save_map.h"
#include "na_mapping/rtk_pos_raw.h"
#include "na_mapping/rtk_heading_raw.h"

#include "MapOptimization.hpp"
#include <GeographicLib/UTMUPS.hpp>

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
int    add_point_size = 0, kdtree_delete_counter = 0;
bool   pcd_save_en = false, time_sync_en = false, extrinsic_est_en = true, path_en = true;
/**************************/

float res_last[100000] = {0.0};
float DET_RANGE = 300.0f;
const float MOV_THRESHOLD = 1.5f;
double time_diff_lidar_to_imu = 0.0;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string map_file_path, lid_topic, imu_topic;

double last_timestamp_lidar = 0, last_timestamp_imu = -1.0,last_timestamp_leg= -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0, fov_deg = 0;
double cube_len = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    scan_count = 0, publish_count = 0;
int    feats_down_size = 0, NUM_MAX_ITERATIONS = 0, pcd_save_interval = -1, pcd_index = 0;

bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<BoxPointType> cub_needrm;
vector<PointVector>  Nearest_Points; 
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
deque<nav_msgs::Odometry::ConstPtr> leg_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());          //畸变纠正后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());         //畸变纠正后降采样的单帧点云，W系

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;

V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;

esekfom::esekf kf;

state_ikfom state_point;

Eigen::Vector3d pos_lid;  //估计的W系下的位置

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());


ros::ServiceServer srvSaveMap;

float mappingSurfLeafSize;

string leg_topic;
bool useleg;



void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

bool first_lidar = false;
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    mtx_buffer.lock();

    if(!first_lidar)
        first_lidar = true;

    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    // time_buffer.push_back(msg->header.stamp.toSec());
    // cout<<"lidar time begin:"<<ptr->points.front().curvature<<endl;
    // cout<<"lidar time end:"<<ptr->points.back().curvature<<endl;
    time_buffer.push_back(msg->header.stamp.toSec()-ptr->points.back().curvature/1000.0);//因为msg的时间戳是雷达帧结束时间，所以要转成开始时间.
    last_timestamp_lidar = msg->header.stamp.toSec();
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) 
{
    mtx_buffer.lock();
    if(!first_lidar)
        first_lidar = true;
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(msg->header.stamp.toSec());
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}





double timediff_lidar_wrt_imu = 0.0;

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    publish_count ++;
    // cout<<"IMU got at: "<<msg_in->header.stamp.toSec()<<endl;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() - time_diff_lidar_to_imu);

    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, clear buffer");
        imu_buffer.clear();
        imu_rtk_buffer.clear();
    }

    last_timestamp_imu = timestamp;

    imu_buffer.push_back(msg);
    imu_rtk_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void leg_cbk(const nav_msgs::Odometry::ConstPtr &msg_in)
{
    nav_msgs::Odometry::Ptr msg(new nav_msgs::Odometry(*msg_in));
    double timestamp = msg->header.stamp.toSec();

    mtx_buffer.lock();

    if (timestamp < last_timestamp_leg)
    {
        ROS_WARN("leg loop back, clear buffer");
        leg_buffer.clear();
    }

    last_timestamp_leg = timestamp;

    leg_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    
}

double lidar_mean_scantime = 0.0;
int    scan_num = 0;
/* last_time_packed是否初始化 */
bool initialized_last_time_packed = false;
double last_time_packed=-1;
/* 数据打包的时间间隔,如果以某个传感器为打包断点,那么时间间隔就是传感器频率的倒数,unit:s */
double time_interval_packed = 0.1;//0.1
/* lidar数据异常(没有收到数据)情况下的最大等待时间,对于低频传感器通常设置为帧间隔时间的一半,
* 超过该时间数据还没到来,就认为本次打包该传感器数据异常,unit:s */
double time_wait_max_lidar = 0.05;

//把当前要处理的LIDAR和IMU数据打包到meas
bool sync_packages(MeasureGroup &meas)
{

    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!first_lidar)
    {
        return false;
    }

    if(!lidar_pushed)
    {
        if(!lidar_buffer.empty())
        {
            meas.lidar = lidar_buffer.front();
            meas.lidar_beg_time = time_buffer.front();

            last_lidar_end_time = lidar_end_time;

            if (meas.lidar->points.size() <= 5) // time too little
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
                ROS_WARN("Too few input point cloud!\n");
            }
            else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
            {
                lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            }
            else
            {
                scan_num ++;
                lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
                lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
            }

            meas.lidar_end_time = lidar_end_time;
            meas.lidar_vaild = true;
            lidar_pushed = true;
            meas.package_end_time = lidar_end_time;
        }
        
    }
    /* 如果lidar还没取出 */
    if (!lidar_pushed)
    return false;

    // if (last_timestamp_imu < lidar_end_time)
    // {
    //     return false;
    // }
    
    //确保具备提取imu数据的条件，即最后一个imu时间大于包的结束时间
    if (last_timestamp_imu <= meas.package_end_time)
        return false;

    //确保具备提取leg数据的条件，即最后一个leg时间大于包的结束时间
    if(useleg)
    {
        if (last_timestamp_leg <= meas.package_end_time)
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    // while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    while ((!imu_buffer.empty()) && (imu_time < meas.package_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        // if(imu_time > lidar_end_time) break;
        if(imu_time > meas.package_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    /*** push leg data, and pop from leg buffer ***/
    if(!leg_buffer.empty() && useleg)
    {
        double leg_time = leg_buffer.front()->header.stamp.toSec();
        meas.leg.clear();
        while ((!leg_buffer.empty()) && (leg_time < lidar_end_time))
        {
            leg_time = leg_buffer.front()->header.stamp.toSec();
            if(leg_time > lidar_end_time) break;
            meas.leg.push_back(leg_buffer.front());
            leg_buffer.pop_front();
        }
    }
    
    //判断leg数据是否有效
    if(!meas.leg.empty())
    {
        meas.leg_vaild = true;
    }
    else
    {
        meas.leg_vaild = false;
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;

    last_time_packed = meas.package_end_time;
    return true;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix()*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot.matrix() * (state_point.offset_R_L_I.matrix()*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}


BoxPointType LocalMap_Points;           // ikd-tree地图立方体的2个角点
bool Localmap_Initialized = false;      // 局部地图是否初始化
void lasermap_fov_segment()      //todo 维护ikd-tree 一个盒子类型地图边界
{
    cub_needrm.clear();     // 清空需要移除的区域
    kdtree_delete_counter = 0;

    V3D pos_LiD = pos_lid;  // W系下位置
    //初始化局部地图范围，以pos_LiD为中心,长宽高均为cube_len
    if (!Localmap_Initialized){
        for (int i = 0; i < 3; i++){
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }

    //各个方向上pos_LiD与局部地图边界的距离
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++){
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        // 与某个方向上的边界距离（1.5*300m）太小，标记需要移除need_move(FAST-LIO2论文Fig.3)
        // if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
        if (dist_to_map_edge[i][0] <= 60 || dist_to_map_edge[i][1] <= 60) need_move = true; //todo
    }
    if (!need_move) return;  //如果不需要，直接返回，不更改局部地图

    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    //需要移动的距离
    // float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    float mov_dist = 20.0;
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        // if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
        if (dist_to_map_edge[i][0] <= 20){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        // } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
        } else if (dist_to_map_edge[i][1] <= 20){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);

    if(cub_needrm.size() > 0) kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm); //删除指定范围内的点
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I.matrix()*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

//根据最新估计位姿  增量添加点云到map
void map_incremental()
{
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);
    for (int i = 0; i < feats_down_size; i++)
    {
        //转换到世界坐标系
        pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
    
        if (!Nearest_Points[i].empty() && flg_EKF_inited)
        {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType mid_point;   //点所在体素的中心
            mid_point.x = floor(feats_down_world->points[i].x/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z/filter_size_map_min)*filter_size_map_min + 0.5 * filter_size_map_min;
            float dist  = calc_dist(feats_down_world->points[i],mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min && fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min && fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min){
                PointNoNeedDownsample.push_back(feats_down_world->points[i]); //如果距离最近的点都在体素外，则该点不需要Downsample
                continue;
            }
            for (int j = 0; j < NUM_MATCH_POINTS; j ++)
            {
                if (points_near.size() < NUM_MATCH_POINTS) break;
                if (calc_dist(points_near[j], mid_point) < dist)  //如果近邻点距离 < 当前点距离，不添加该点
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.push_back(feats_down_world->points[i]);
        }
        else
        {
            PointToAdd.push_back(feats_down_world->points[i]);
        }
    }

    double st_time = omp_get_wtime();
    add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    add_point_size = PointToAdd.size() + PointNoNeedDownsample.size();
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());
void publish_frame_world(const ros::Publisher & pubLaserCloudFull_)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI::Ptr laserCloudWorld( \
                        new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++)
        {
            pointBodyToWorld(&laserCloudFullRes->points[i], \
                                &laserCloudWorld->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull_.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }
}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);

    auto q_ = Eigen::Quaterniond(state_point.rot.matrix());
    out.pose.orientation.x = q_.coeffs()[0];
    out.pose.orientation.y = q_.coeffs()[1];
    out.pose.orientation.z = q_.coeffs()[2];
    out.pose.orientation.w = q_.coeffs()[3];   
}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);

    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

//发布更新后的轨迹
void publish_path_update(const ros::Publisher pubPath)
{
    ros::Time timeLaserInfoStamp = ros::Time().fromSec(lidar_end_time); //  时间戳
    string odometryFrame = "camera_init";
    if (pubPath.getNumSubscribers() != 0)
    {
        /*** if path is too large, the rvis will crash ***/
        static int kkk = 0;
        kkk++;
        if (kkk % 10 == 0)
        {
            // path.poses.push_back(globalPath);
            globalPath.header.stamp = timeLaserInfoStamp;
            globalPath.header.frame_id = odometryFrame;
            pubPath.publish(globalPath);
        }
    }
}

//  发布gnss 轨迹
void publish_gnss_path(const ros::Publisher pubPath)
{
    gps_path.header.stamp = ros::Time().fromSec(lidar_end_time);
    gps_path.header.frame_id = "camera_init";

    static int jjj = 0;
    jjj++;
    if (jjj % 10 == 0) 
    {
        pubPath.publish(gps_path);
    }
}

bool saveMapService(na_mapping::save_mapRequest& req, na_mapping::save_mapResponse& res)
{
    cout<<"start save map"<<endl;

    pcl::PointCloud<PointType>::Ptr MapKeyPosesDS(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr MapKeyFrames(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr MapKeyFramesDS(new pcl::PointCloud<PointType>());

    pcl::VoxelGrid<PointType> downSizeFilterMapKeyPoses;
    downSizeFilterMapKeyPoses.setLeafSize(globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity, globalMapVisualizationPoseDensity);
    downSizeFilterMapKeyPoses.setInputCloud(cloudKeyPoses3D);
    downSizeFilterMapKeyPoses.filter(*MapKeyPosesDS);

    for (int i = 0; i < (int)MapKeyPosesDS->size(); ++i)
    {
        int thisKeyInd = (int)MapKeyPosesDS->points[i].intensity;
        *MapKeyFrames += *transformPointCloud(surfCloudKeyFrames[thisKeyInd], &cloudKeyPoses6D->points[thisKeyInd],state_point); //  fast_lio only use  surfCloud
    }

    pcl::VoxelGrid<PointType> downSizeFilterGlobalMapKeyFrames;                                                                                   // for global map visualization
    // downSizeFilterGlobalMapKeyFrames.setLeafSize(globalMapVisualizationLeafSize, globalMapVisualizationLeafSize, globalMapVisualizationLeafSize); // for global map visualization
    // downSizeFilterGlobalMapKeyFrames.setLeafSize(0.2, 0.2, 0.2);
    downSizeFilterGlobalMapKeyFrames.setLeafSize(0.4, 0.4, 0.4);
    downSizeFilterGlobalMapKeyFrames.setInputCloud(MapKeyFrames);
    downSizeFilterGlobalMapKeyFrames.filter(*MapKeyFramesDS);

    pcl::io::savePCDFileBinary(savemappath,*MapKeyFramesDS);

    if (usertk==true)
    {
        FILE *fp;
        fp=fopen(saveposepath.c_str(),"w+");
        fprintf(fp,"%2.10lf %3.10lf %2.5lf",lat0,lon0,alt0);
        fclose(fp); 
    }

    cout<<"save map done"<<endl;
    res.success = true;
    return true;
}

void gnss_cbk(const na_mapping::rtk_pos_raw::ConstPtr& msg_in)
{
    //判断是否使用rtk融合
    if (usertk==false)
        return;
    //判断gps是否有效
    if (msg_in->pos_type!=50)
        return;

    if (msg_in->hgt_std_dev > 0.05)
        return;

    int zone;
    bool northp;

    //初始化utm坐标系原点
    if(!rtk_p0_init)
    {
        lat0 = msg_in->lat;
        lon0 = msg_in->lon;
        alt0 = msg_in->hgt;   

        GeographicLib::UTMUPS::Forward(lat0, lon0, zone, northp, utm_x0, utm_y0);     
        utm_z0 = alt0;
        rtk_p0_init = true;
    }

    //  ROS_INFO("GNSS DATA IN ");
    double timestamp = msg_in->header.stamp.toSec()+rtk_time_grift;

    mtx_buffer.lock();

    // 没有进行时间纠正
    if (timestamp < last_timestamp_gnss)
    {
        ROS_WARN("gnss loop back, clear buffer");
        gnss_buffer.clear();
    }

    last_timestamp_gnss = timestamp;

    // convert ROS NavSatFix to GeographicLib compatible GNSS message:
    gnss_data.time = msg_in->header.stamp.toSec()+rtk_time_grift;
    gnss_data.status = msg_in->pos_type;
    gnss_data.service = 0;

    //设置rtk协方差
    double posecov;
    posecov=0.05*0.05;

    gnss_data.pose_cov[0] = posecov;
    gnss_data.pose_cov[1] = posecov;
    gnss_data.pose_cov[2] = 4.0*posecov;

    mtx_buffer.unlock();
    
    double utm_x,utm_y,utm_z;
    GeographicLib::UTMUPS::Forward(msg_in->lat, msg_in->lon, zone, northp, utm_x, utm_y);
    utm_z = msg_in->hgt;
    utm_x -= utm_x0;
    utm_y -= utm_y0;
    utm_z -= utm_z0;

    nav_msgs::Odometry gnss_data_enu ;
    gnss_data_enu.header.stamp = ros::Time().fromSec(gnss_data.time);
    gnss_data_enu.pose.pose.position.x =  utm_x ;  //东
    gnss_data_enu.pose.pose.position.y =  utm_y ;  //北
    gnss_data_enu.pose.pose.position.z =  utm_z ;  //天

    gnss_data_enu.pose.covariance[0] = gnss_data.pose_cov[0] ;
    gnss_data_enu.pose.covariance[7] = gnss_data.pose_cov[1] ;
    gnss_data_enu.pose.covariance[14] = gnss_data.pose_cov[2] ;

    gnss_buffer.push_back(gnss_data_enu);
    // visial gnss path in rviz:
    msg_gnss_pose.header.frame_id = "camera_init";
    msg_gnss_pose.header.stamp = ros::Time().fromSec(gnss_data.time);

    Eigen::Matrix4d gnss_pose = Eigen::Matrix4d::Identity();

    gnss_pose(0,3) = utm_x ;
    gnss_pose(1,3) = utm_y ;
    gnss_pose(2,3) = utm_z ;

    msg_gnss_pose.pose.position.x = gnss_pose(0,3) ;  
    msg_gnss_pose.pose.position.y = gnss_pose(1,3) ;
    msg_gnss_pose.pose.position.z = gnss_pose(2,3) ;

    gps_path.poses.push_back(msg_gnss_pose); 
    rtk_time.push_back(msg_in->header.stamp.toSec());
}
// void gnss_cbk(const na_mapping::rtk_pos_raw::ConstPtr& msg_in)
// {
//     //判断是否使用rtk融合
//     if (usertk==false)
//         return;
//     //判断gps是否有效
//     if (msg_in->pos_type!=50)
//         return;

//     if (msg_in->hgt_std_dev > 0.05)
//         return;

//     //初始化局部东北天坐标系原点
//     if(use_rtk_heading && !rtk_p0_init)
//     {
//         if(rtk_heading_vaild)
//         {
//             lat0 = msg_in->lat;
//             lon0 = msg_in->lon;
//             alt0 = msg_in->hgt;
//             gnss_data.InitOriginPosition(lat0, lon0, alt0) ; 
            
//             //计算初始雷达的位置（经纬高）
//             // Eigen::Vector3d pose_Lidar_t0 = Vector3d::Zero();
//             // Eigen::AngleAxisd rot_z_btol(rtk_heading, Eigen::Vector3d::UnitZ());
//             // Eigen::Matrix3d dog_attitude = rot_z_btol.matrix();
//             // pose_Lidar_t0 = - dog_attitude * rtk_T_wrt_Lidar;
//             // gnss_data.InitOriginPosition(msg_in->lat, msg_in->lon, msg_in->hgt) ;
//             // gnss_data.Reverse(pose_Lidar_t0[0],pose_Lidar_t0[1],pose_Lidar_t0[2],lat0,lon0,alt0);
//             // gnss_data.InitOriginPosition(lat0, lon0, alt0) ; 

//             rtk_p0_init = true;
//         }
//         else
//         {
//             return;
//         }
//     }
//     //  ROS_INFO("GNSS DATA IN ");
//     double timestamp = msg_in->header.stamp.toSec()+rtk_time_grift;

//     mtx_buffer.lock();

//     // 没有进行时间纠正
//     if (timestamp < last_timestamp_gnss)
//     {
//         ROS_WARN("gnss loop back, clear buffer");
//         gnss_buffer.clear();
//     }

//     last_timestamp_gnss = timestamp;

//     // convert ROS NavSatFix to GeographicLib compatible GNSS message:
//     gnss_data.time = msg_in->header.stamp.toSec()+rtk_time_grift;
//     gnss_data.status = msg_in->pos_type;
//     gnss_data.service = 0;

//     //通过gps_qual给出协方差
//     double posecov;
//     posecov=0.05*0.05;
//     // posecov=0.01*0.01;
//     // if (gnss_data.status == 4)
//     //     posecov=0.05*0.05;
//     // else if (gnss_data.status == 5)
//     //     posecov=1.0*1.0;
//     // else if (gnss_data.status == 1)
//     //     posecov=10.0*10.0;
//     // else
//     //     return;

//     gnss_data.pose_cov[0] = posecov;
//     gnss_data.pose_cov[1] = posecov;
//     gnss_data.pose_cov[2] = 2.0*posecov;

//     mtx_buffer.unlock();
   
//     if(!gnss_inited){           //  初始化位置
//         // gnss_data.InitOriginPosition(msg_in->lat, msg_in->lon, msg_in->hgt) ; 
//         // lat0=msg_in->lat;
//         // lon0=msg_in->lon;
//         // alt0=msg_in->hgt;
//         gnss_inited = true ;
//     }else{                               
//         //经纬高转东北天
//         gnss_data.UpdateXYZ(msg_in->lat, msg_in->lon, msg_in->hgt) ;             
//         nav_msgs::Odometry gnss_data_enu ;
//         // add new message to buffer:
//         gnss_data_enu.header.stamp = ros::Time().fromSec(gnss_data.time);

//         // V3D dp;
//         // dp = state_point.rot.matrix()*rtk_T_wrt_Lidar;

//         gnss_data_enu.pose.pose.position.x =  gnss_data.local_E ;  //东
//         gnss_data_enu.pose.pose.position.y =  gnss_data.local_N ;  //北
//         gnss_data_enu.pose.pose.position.z =  gnss_data.local_U ;  //天

//         // gnss_data_enu.pose.pose.orientation.x =  geoQuat.x ;                //  gnss 的姿态不可观，所以姿态只用于可视化，取自imu
//         // gnss_data_enu.pose.pose.orientation.y =  geoQuat.y;
//         // gnss_data_enu.pose.pose.orientation.z =  geoQuat.z;
//         // gnss_data_enu.pose.pose.orientation.w =  geoQuat.w;

//         gnss_data_enu.pose.covariance[0] = gnss_data.pose_cov[0] ;
//         gnss_data_enu.pose.covariance[7] = gnss_data.pose_cov[1] ;
//         gnss_data_enu.pose.covariance[14] = gnss_data.pose_cov[2] ;

//         gnss_buffer.push_back(gnss_data_enu);

//         // visial gnss path in rviz:
//         msg_gnss_pose.header.frame_id = "camera_init";
//         msg_gnss_pose.header.stamp = ros::Time().fromSec(gnss_data.time);
//         // Eigen::Vector3d gnss_pose_ (gnss_data.local_E, gnss_data.local_N, - gnss_data.local_U); 
//         // Eigen::Vector3d gnss_pose_ (gnss_data.local_N, gnss_data.local_E, - gnss_data.local_U); 
//         Eigen::Matrix4d gnss_pose = Eigen::Matrix4d::Identity();

//         // V3D dp;
//         // dp=state_point.rot.matrix()*rtk_T_wrt_Lidar;

//         gnss_pose(0,3) = gnss_data.local_E ;
//         gnss_pose(1,3) = gnss_data.local_N ;
//         gnss_pose(2,3) = gnss_data.local_U ;

//         // gnss_pose(0,3) = gnss_data.local_E - dp[0];
//         // gnss_pose(1,3) = gnss_data.local_N - dp[1];
//         // gnss_pose(2,3) = gnss_data.local_U - dp[2];

//         // Eigen::Isometry3d gnss_to_lidar(Gnss_R_wrt_Lidar) ;
//         // gnss_to_lidar.pretranslate(Gnss_T_wrt_Lidar);
//         // gnss_pose  =  gnss_to_lidar  *  gnss_pose ;                    //  gnss 转到 lidar 系下

//         msg_gnss_pose.pose.position.x = gnss_pose(0,3) ;  
//         msg_gnss_pose.pose.position.y = gnss_pose(1,3) ;
//         msg_gnss_pose.pose.position.z = gnss_pose(2,3) ;

//         gps_path.poses.push_back(msg_gnss_pose); 
//     }
// }

void gnss_heading_cbk(const na_mapping::rtk_heading_raw::ConstPtr& msg_in)
{
    //如果不是固定解或不使用rtk的航向,return;
    if(msg_in->pos_type != 50 || use_rtk_heading == false)
    {
        return ;
    }
    
    // rtk_heading = (180.0 - msg_in->heading) * M_PI / 180.0;
    rtk_heading =  - msg_in->heading * M_PI / 180.0;
    rtk_heading_vaild = true;
}


//todo read 主函数
int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);                // 是否发布当前正在扫描的点云的topic
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);              // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic 
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);     // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic，需要该变量和上一个变量同时为true才发布
    nh.param<int>("max_iteration",NUM_MAX_ITERATIONS,4);                        // 卡尔曼滤波的最大迭代次数
    nh.param<string>("map_file_path",map_file_path,"");                         // 地图保存路径
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");              // 雷达点云topic名称
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");               // IMU的topic名称
    nh.param<bool>("common/time_sync_en", time_sync_en, false);                 // 是否需要时间同步，只有当外部未进行时间同步时设为true
    nh.param<double>("common/time_offset_lidar_to_imu", time_diff_lidar_to_imu, 0.0);
    nh.param<double>("filter_size_corner",filter_size_corner_min,0.5);          // VoxelGrid降采样时的体素大小
    nh.param<double>("filter_size_surf",filter_size_surf_min,0.5);
    nh.param<double>("filter_size_map",filter_size_map_min,0.5);
    nh.param<double>("cube_side_length",cube_len,200);                          // 地图的局部区域的长度（FastLio2论文中有解释）
    nh.param<float>("mapping/det_range",DET_RANGE,300.f);                       // 激光雷达的最大探测范围
    nh.param<double>("mapping/fov_degree",fov_deg,180);
    nh.param<double>("mapping/gyr_cov",gyr_cov,0.1);                            // IMU陀螺仪的协方差
    nh.param<double>("mapping/acc_cov",acc_cov,0.1);                            // IMU加速度计的协方差
    nh.param<double>("mapping/b_gyr_cov",b_gyr_cov,0.0001);                     // IMU陀螺仪偏置的协方差
    nh.param<double>("mapping/b_acc_cov",b_acc_cov,0.0001);                     // IMU加速度计偏置的协方差
    nh.param<double>("preprocess/blind", p_pre->blind, 0.01);                   // 最小距离阈值，即过滤掉0～blind范围内的点云
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);            // 激光雷达的类型
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);                  // 激光雷达扫描的线数（livox avia为6线）
    nh.param<int>("preprocess/timestamp_unit", p_pre->time_unit, US);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("point_filter_num", p_pre->point_filter_num, 2);              // 采样间隔，即每隔point_filter_num个点取1个点
    nh.param<bool>("feature_extract_enable", p_pre->feature_enabled, false);    // 是否提取特征点（FAST_LIO2默认不进行特征点提取）
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>()); // 雷达相对于IMU的外参T（即雷达在IMU坐标系中的坐标）
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>()); // 雷达相对于IMU的外参R
    

    // save keyframes
    nh.param<float>("surroundingkeyframeAddingDistThreshold", surroundingkeyframeAddingDistThreshold, 1.0); //判断是否为关键帧的距离阈值(m)
    nh.param<float>("surroundingkeyframeAddingAngleThreshold", surroundingkeyframeAddingAngleThreshold, 0.2); //判断是否为关键帧的角度阈值(rad)
    // Visualization
    nh.param<float>("globalMapVisualizationSearchRadius", globalMapVisualizationSearchRadius, 50); //重构ikd树的搜索范围(m)
    nh.param<float>("globalMapVisualizationPoseDensity", globalMapVisualizationPoseDensity, 2.0);  //重构ikd树对关键帧位置的降采样体素大小
    nh.param<float>("globalMapVisualizationLeafSize", globalMapVisualizationLeafSize, 0.4);        //重构ikd树的降采样体素大小
    // loop clousre
    nh.param<bool>("loopClosureEnableFlag", loopClosureEnableFlag, true); //是否加入回环因子
    nh.param<float>("loopClosureFrequency", loopClosureFrequency, 1.0);   //回环检测的频率
    nh.param<float>("historyKeyframeSearchRadius", historyKeyframeSearchRadius, 20.0); //回环检测的搜索范围(m)
    nh.param<float>("historyKeyframeSearchTimeDiff", historyKeyframeSearchTimeDiff, 30.0); //回环检测的时间阈值(s)
    nh.param<float>("historyKeyframeFitnessScore", historyKeyframeFitnessScore, 0.3); //回环检测icp匹配的分数阈值
    nh.param<int>("historyKeyframeSearchNum", historyKeyframeSearchNum, 25);    //回环检测局部地图的使用的相邻关键帧数量
    nh.param<float>("mappingSurfLeafSize", mappingSurfLeafSize, 0.2);           //回环检测局部地图的降采样体素大小
    //rtk
    nh.param<bool>("usertk", usertk, false); //是否加入gps因子
    nh.param<bool>("use_rtk_heading", use_rtk_heading, false); //是否使用rtk航向初始化
    nh.param<string>("common/gnss_topic", gnss_topic,"/rtk_pos_raw");   //gps的topic名称
    nh.param<string>("common/gnss_heading_topic", gnss_heading_topic,"/rtk_heading_raw");   //gps的topic名称
    nh.param<float>("gpsCovThreshold", gpsCovThreshold, 0.2);           //gps的协方差阈值
    nh.param<float>("poseCovThreshold", poseCovThreshold, 0.01);        //位姿的协方差阈值，过小就不用加入gps因子
    nh.param<bool>("useGpsElevation", useGpsElevation, true);           //是否使用gps的高度信息
    nh.param<vector<double>>("mapping/rtk2Lidar_T", rtk2Lidar_T, vector<double>()); // rtk相对于雷达的外参T（即rtk在Lidar坐标系中的坐标）

    nh.param<int>("numberOfCores", numberOfCores, 2);                   //使用的cpu核数
    nh.param<bool>("recontructKdTree", recontructKdTree, true);         //是否重构ikd树
    nh.param<std::string>("savemappath", savemappath, "/home/ywb/s-fast-lio/src/S-FAST_LIO/PCD/cloud_map.pcd"); //保存地图点云的路径
    nh.param<std::string>("saveposepath", saveposepath, "/home/ywb/s-fast-lio/src/S-FAST_LIO/PCD/pose.txt");    //保存地图原点的位置（只有用gps的时候会保存）
  
    nh.param<string>("common/leg_topic", leg_topic,"/leg_odom");   //leg的topic名称
    nh.param<bool>("useleg",useleg,false); //是否使用leg
    cout<<"Lidar_type: "<<p_pre->lidar_type<<endl;
    // 初始化path的header（包括时间戳和帧id），path用于保存odemetry的路径
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    // ISAM2参数
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    isam = new gtsam::ISAM2(parameters);

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk,ros::TransportHints().unreliable());
    ros::Subscriber sub_gnss = nh.subscribe(gnss_topic, 200000, gnss_cbk); //gnss
    ros::Subscriber sub_gnss_heading = nh.subscribe(gnss_heading_topic, 200000, gnss_heading_cbk); 
    ros::Subscriber sub_leg = nh.subscribe(leg_topic, 200000, leg_cbk,ros::TransportHints().unreliable()); 
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> ("/Odometry", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> ("/path", 100000);
    ros::Publisher pubPathUpdate = nh.advertise<nav_msgs::Path>("s_fast_lio/path_update", 100000);                   //  isam更新后的path
    ros::Publisher pubGnssPath = nh.advertise<nav_msgs::Path>("/gnss_path", 100000);
    // 发布闭环边，rviz中表现为闭环帧之间的连线
    pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/fast_lio_sam/mapping/loop_closure_constraints", 1);

    srvSaveMap  = nh.advertiseService("/save_map" ,  &saveMapService); // 保存地图服务

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    downSizeFilterICP.setLeafSize(mappingSurfLeafSize, mappingSurfLeafSize, mappingSurfLeafSize);


    shared_ptr<ImuProcess> p_imu1(new ImuProcess());
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    rtk_T_wrt_Lidar<<VEC_FROM_ARRAY(rtk2Lidar_T);
    p_imu1->set_param(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, acc_cov), 
                        V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov), V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    // 回环检测线程
    std::thread loopthread(loopClosureThread, &lidar_end_time, &state_point);

    Eigen::Matrix3d Sigma_leg = Eigen::Matrix3d::Identity(); //leg里程计的协方差
    Sigma_leg(0, 0) = 0.01;//0.001
    Sigma_leg(1, 1) = 0.01;
    Sigma_leg(2, 2) = 0.01;

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);

    while (ros::ok())
    {
        if (flg_exit) break;
        ros::spinOnce();

        if(sync_packages(Measures))  //把一次的IMU和LIDAR数据打包到Measures
        {   
            //使用rtk航向信息初始化姿态角
            if(use_rtk_heading && !rtk_heading_init)
            {
                if(rtk_heading_vaild)
                {
                    //姿态初始化
                    Eigen::AngleAxisd rot_z_btol(rtk_heading, Eigen::Vector3d::UnitZ());
                    Eigen::Matrix3d dog_attitude = rot_z_btol.matrix();
                    state_point.rot=Sophus::SO3d(dog_attitude);
                    
                    //位置初始化
                    Eigen::Vector3d pose_Lidar_t0 = Vector3d::Zero();
                    pose_Lidar_t0 = - dog_attitude * rtk_T_wrt_Lidar;
                    state_point.pos = pose_Lidar_t0;

                    rtk_heading_init = true;
                    kf.change_x(state_point);
                }
                else
                {
                    rate.sleep();
                    continue;
                }
            }

            double t00 = omp_get_wtime();

            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu1->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }

            state_point_last = state_point;

            p_imu1->Process(Measures, kf, feats_undistort);  

            //如果feats_undistort为空 ROS_WARN
            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

            lasermap_fov_segment();     //更新localmap边界，然后降采样当前帧点云

            //点云下采样
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            feats_down_size = feats_down_body->points.size();
            // cout<<"feats_down_size:"<<feats_down_size<<endl;

            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            //初始化ikdtree(ikdtree为空时)
            if(ikdtree.Root_Node == nullptr)
            {
                ikdtree.set_downsample_param(filter_size_map_min);
                feats_down_world->resize(feats_down_size);
                for(int i = 0; i < feats_down_size; i++)
                {
                    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));  //lidar坐标系转到世界坐标系
                }
                ikdtree.Build(feats_down_world->points);        //根据世界坐标系下的点构建ikdtree 
                continue;
            }
            
            //是否发布ikdtree点云

            if(1) 
            {
                PointVector ().swap(ikdtree.PCL_Storage);
                ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
                featsFromMap->clear();
                featsFromMap->points = ikdtree.PCL_Storage;
            }

            Eigen::Vector3d z_leg = Eigen::Vector3d::Zero();
            if(Measures.leg_vaild == true)
            {
                nav_msgs::Odometry::ConstPtr leg_back = Measures.leg.back();
                //z_leg只有前向和左向的速度
                z_leg(0)=leg_back->twist.twist.linear.x;
                z_leg(1)=leg_back->twist.twist.linear.y;
            }
            
            /*** iterated state estimation ***/
            Nearest_Points.resize(feats_down_size);         //存储近邻点的vector
            // kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_body, ikdtree, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en);
            kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_body, ikdtree, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en,
                                                  Sigma_leg,z_leg,useleg,Measures.leg_vaild,Measures.lidar_vaild);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

            /***后端融合***/
            getCurPose(state_point);
            /*back end*/
            // 1.计算当前帧与前一帧位姿变换，如果变化太小，不设为关键帧，反之设为关键帧
            // 2.添加激光里程计因子、GPS因子、闭环因子
            // 3.执行因子图优化
            // 4.得到当前帧优化后的位姿，位姿协方差
            // 5.添加cloudKeyPoses3D，cloudKeyPoses6D，更新transformTobeMapped，添加当前关键帧的角点、平面点集合
            saveKeyFramesAndFactor(lidar_end_time,kf,state_point,feats_down_body,feats_undistort);
            // 更新因子图中所有变量节点的位姿，也就是所有历史关键帧的位姿，更新里程计轨迹， 重构ikdtree
            correctPoses(state_point, ikdtree);

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);

            /*** add the feature points to map kdtree ***/
            feats_down_world->resize(feats_down_size);

            double time_add_map1 = omp_get_wtime();
            map_incremental();
            double time_add_map2 = omp_get_wtime();
            // cout<<"feats_down_size:"<<feats_down_size<<endl;
            // cout<<"ikdtree size:"<<ikdtree.validnum()<<endl;
            // cout<<"add points time:"<<time_add_map2 - time_add_map1<<endl;
            
            /******* Publish points *******/
            if (path_en)
            {
                publish_path(pubPath);
                publish_path_update(pubPathUpdate);             //   发布经过isam2优化后的路径
                publish_gnss_path(pubGnssPath);                        //   发布gnss轨迹
            }                         
            if (scan_pub_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);

            publish_map(pubLaserCloudMap);

            double t11 = omp_get_wtime();

            //输出航向角测试
            // Eigen::Vector3d eulerAngle = state_point.rot.matrix().eulerAngles(2,1,0);        //  yaw pitch roll  单位：弧度
            // cout<<"heading:"<<eulerAngle(0)*180/M_PI<<endl;
            // double yaw = atan2(state_point.rot.matrix()(1,0),state_point.rot.matrix()(0,0)) * 180 / M_PI;
            // cout<<"heading:"<<yaw<<endl;
        }

        rate.sleep();
    }
    
    //输出轨迹
    if(0)
    {
        string file_name1 = "/home/ywb/NR_robot/na_mapping/src/na_mapping/Path/lidar_path.txt";
        string file_name2 = "/home/ywb/NR_robot/na_mapping/src/na_mapping/Path/rtk_path.txt";
        ofstream fout;
        fout.open(file_name1);
        cout<<"globalPath size:"<<globalPath.poses.size()<<endl;
        cout<<"keyframe_time size:"<<keyframe_time.size()<<endl;


        for(int i=0;i<globalPath.poses.size();i++)
        {   
            //计算rtk的位置
            V3D pos_imu,pos_rtk;
            Eigen::Quaterniond q_imu;
            
            pos_imu << globalPath.poses[i].pose.position.x,
                    globalPath.poses[i].pose.position.y,
                    globalPath.poses[i].pose.position.z;

            q_imu.w() = globalPath.poses[i].pose.orientation.w;
            q_imu.x() = globalPath.poses[i].pose.orientation.x;
            q_imu.y() = globalPath.poses[i].pose.orientation.y;
            q_imu.z() = globalPath.poses[i].pose.orientation.z;

            pos_rtk = pos_imu + q_imu.normalized().toRotationMatrix() * rtk_T_wrt_Lidar;

            fout << fixed << setprecision(9) << keyframe_time[i] << " "
                << fixed << setprecision(4) << pos_rtk[0] << " "
                << fixed << setprecision(4) << pos_rtk[1] << " "
                << fixed << setprecision(4) << pos_rtk[2] << endl;
        }
        fout.close();

        fout.open(file_name2);
        cout<<"rtk_time size:"<<rtk_time.size()<<endl;
        cout<<"gpspath size:"<<gps_path.poses.size()<<endl;
        for(int i=0;i<gps_path.poses.size();i++)
        {   
            fout << fixed << setprecision(9) << rtk_time[i] << " "
                << fixed << setprecision(4) << gps_path.poses[i].pose.position.x << " "
                << fixed << setprecision(4) << gps_path.poses[i].pose.position.y << " "
                << fixed << setprecision(4) << gps_path.poses[i].pose.position.z << endl;
        }
        fout.close();
    }


    startFlag = false;
    loopthread.join(); //  分离线程

    return 0;
}
