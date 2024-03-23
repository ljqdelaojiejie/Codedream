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
#include <pcl/registration/ndt.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
// #include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include <ikd-Tree/ikd_Tree.h>
#include<Eigen/Dense>
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

#include "na_localization/rtk_pos_raw.h"
#include "na_localization/rtk_heading_raw.h"
#include "na_localization/sensor_vaild.h"

#include <std_msgs/Bool.h>
#include <std_msgs/Int32.h>

#include <GeographicLib/UTMUPS.hpp>

#include "map_management.h"

#include "alg/common/common.h"
#include "alg/relocService/relocService.h"
#include "plugins/reloc_plugin/reloc_plugin.h"
//todo 添加bfs搜索使用的头文件

// #include "matchRateCal/match_rate_cal.h"

#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)//0.001
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
deque<double>                     time_end_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_rtk_buffer;
deque<nav_msgs::Odometry::ConstPtr> leg_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());          //畸变纠正后降采样的单帧点云，lidar系
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());         //畸变纠正后降采样的单帧点云，W系

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

KD_TREE<PointType> ikdtree;
KD_TREE<PointType> ikdtree2;



V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

/*** EKF inputs and output ***/
MeasureGroup Measures;

esekfom::esekf kf;

state_ikfom state_point;
state_ikfom state_point_last;//上一时刻的状态
double package_end_time_last = 0;
Eigen::Vector3d pos_lid;  //估计的W系下的位置

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());

/***优化部分相关变量***/
float transformTobeMapped[6]; //  当前帧的位姿(world系下)

vector<pcl::PointCloud<PointType>::Ptr> surfCloudKeyFrames;   // 历史所有关键帧的平面点集合（降采样）

pcl::PointCloud<PointType>::Ptr cloudKeyPoses3D(new pcl::PointCloud<PointType>());         // 历史关键帧位姿（位置）
pcl::PointCloud<PointTypePose>::Ptr cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>()); // 历史关键帧位姿
pcl::PointCloud<PointType>::Ptr copy_cloudKeyPoses3D(new pcl::PointCloud<PointType>());
pcl::PointCloud<PointTypePose>::Ptr copy_cloudKeyPoses6D(new pcl::PointCloud<PointTypePose>());

nav_msgs::Path globalPath;
bool visual_ikdtree;

bool usertk;
string gnss_topic;
string gnss_heading_topic;
double rtk_time_grift = 0;
double last_timestamp_gnss = -1.0 ;
deque<nav_msgs::Odometry> gnss_buffer;
deque<double> gnss_heading_buffer;
deque<double> heading_time_buffer;
geometry_msgs::PoseStamped msg_gnss_pose;
shared_ptr<GnssProcess> p_gnss(new GnssProcess());
GnssProcess gnss_data;
bool gnss_inited = false ;                        //  是否完成gnss初始化
double lat0,lon0,alt0;
nav_msgs::Path gps_path ;

std::mutex mtx;

string leg_topic;
bool useleg;

//重定位
// bool flag_reposition=false;
bool need_relocal = true;
int Reposition_type;
// bool flag_lidar=false;
// bool flag_extpos=false;
bool flag_manualpos=false;
bool flag_rtkpos=false;
bool flag_rtkheading = false;
bool flag_relocbfs = false;
// pcl::PointCloud<PointType>::Ptr ds_pl_orig (new pcl::PointCloud<PointType>());//声明源点云
pcl::PointCloud<PointType>::Ptr pointcloudmap(new pcl::PointCloud<PointType>());//地图点云
// double extposx,extposy,extposz;
Eigen::Vector3d manualpos = Eigen::Vector3d::Zero();
Eigen::Quaterniond ext_q;
string loadmappath; //读取pcd的路径
string loadposepath; //读取pcd的初始经纬高
int map_count = 0;
sensor_msgs::PointCloud2 globalmapmsg;
ros::Publisher pubglobalmap;
double rtk_heading;
double kf_heading;
bool rtk_vaild = false;
bool rtk_heading_vaild = false;
V3D rtk_T_wrt_Lidar(Zero3d);
vector<double> rtk2Lidar_T(3, 0.0);

std::shared_ptr<plugins::RelocPlugin> reloc_plugin_ptr_; // bfs搜索功能指针

//重构ikd树
bool flag_ikdtree_initial=false;
PointType pos_last,pos_now;
pcl::PointCloud<PointType>::Ptr mapikdtree(new pcl::PointCloud<PointType>());//存储新的ikd树点云
bool flag_reikdtree =false;
std::mutex mtx_reikd;
bool map_update = false;
int flag_thread1 = 0, flag_thread2 = 0; //0表示空闲，1表示正在处理ikdtree1,2表示正在处理ikdtree2
int last_ikdtree = 1; //最新更新的ikdtree

//kd树
pcl::KdTreeFLANN<PointType> kdtree1;
pcl::KdTreeFLANN<PointType> kdtree2;

ros::Publisher pubSensorVaild;
ros::Publisher pubLocalizationVaild;

bool imu_vaild = true; //imu是否有效的标志位
double imu_fault_time; //imu故障的时间

map_management MM;
//ikdtree包围盒
BoxPointType LocalMap_Points;           // ikd-tree地图立方体的2个角点
bool Localmap_Initialized = false;      // 局部地图是否初始化

void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

int count_fault=0;
bool first_lidar = false;


// typedef pcl::PointXYZINormal PointType;
// typedef pcl::PointCloud<PointType> PointCloudXYZI;
// PointCloudXYZI pl_full,pl_surf;  //todo 把livox消息类型转换到pcl 再转换到 sensor msg
//                                 //todo 为了能给bfs搜索的接口
sensor_msgs::PointCloud2 trans_cloud;
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
    p_pre->process(msg, ptr); //todo 把livox消息类型转换成pcl

    lidar_buffer.clear();
    time_buffer.clear();
    time_end_buffer.clear();

    lidar_buffer.push_back(ptr);
    time_end_buffer.push_back(msg->header.stamp.toSec()+ptr->points.back().curvature/1000.0);

    time_buffer.push_back(msg->header.stamp.toSec());
    last_timestamp_lidar = msg->header.stamp.toSec();

    mtx_buffer.unlock();
    sig_buffer.notify_all();

    pcl::toROSMsg(*ptr, trans_cloud);
    
}

// sensor_msgs::PointCloud2 overlap_cloud;
void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg) 
{
    //容错测试
    // count_fault++;
    // if(count_fault>=50 && count_fault<=200) return;
    mtx_buffer.lock();
    if(!first_lidar)
        first_lidar = true;
    // scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);

    //如果雷达点云没有数据，直接return
    // if(ptr->points.size() == 0)
    if(ptr->points.size() < 100)
    {
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }
        
    //定位程序需要避免缓存区数据堆积
    lidar_buffer.clear();
    time_buffer.clear();
    time_end_buffer.clear();

    lidar_buffer.push_back(ptr);
    time_end_buffer.push_back(msg->header.stamp.toSec());

    // cout<<"lidar size:"<<ptr->points.size()<<endl;

    time_buffer.push_back(msg->header.stamp.toSec()-ptr->points.back().curvature/1000.0);//因为rslidar的时间戳是雷达帧结束时间，所以要转成开始时间.
    last_timestamp_lidar = msg->header.stamp.toSec();
    // cout<<"lidar buff have data!!!"<<endl;
    mtx_buffer.unlock();
    sig_buffer.notify_all();

    // overlap_cloud = *msg;
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
int imu_fault = 0;
void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) 
{
    // imu_fault++;
    // if(imu_fault>1000 && imu_fault<1800) return ;

    //如果imu无效，则置有效
    if(imu_vaild == false)
    {
        // imu_fault_time
        imu_vaild = true;
        ROS_INFO("IMU falut recovery! recovery time:%lf",msg_in->header.stamp.toSec()-imu_fault_time);
    }

    // publish_count ++;
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
    if(usertk)
    {
        imu_rtk_buffer.push_back(msg);
    } 
    //监视缓存区的大小，大小超出限制就pop掉最一开始的数据
    if(imu_buffer.size()>1000)
    {
        imu_buffer.pop_front();
        // ROS_WARN("imu buffer is too large!");
    }

    if(imu_rtk_buffer.size()>1000)
    {
        imu_rtk_buffer.pop_front();
        // ROS_WARN("imu_rtk buffer is too large!");
    }

    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void leg_cbk(const nav_msgs::Odometry::ConstPtr &msg_in)
{
    if(!useleg) return;
    
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

    if(leg_buffer.size()>1000)
    {
        leg_buffer.pop_front();
        // ROS_WARN("leg buffer is too large!");
    }
    mtx_buffer.unlock();
    sig_buffer.notify_all();
    
}

double lidar_mean_scantime = 0.0;
// int    scan_num = 0;
/* last_time_packed是否初始化 */
bool initialized_last_time_packed = false;
double last_time_packed=-1;
/* 数据打包的时间间隔,如果以某个传感器为打包断点,那么时间间隔就是传感器频率的倒数,unit:s */
double time_interval_packed = 0.1;//0.1
/* lidar数据异常(没有收到数据)情况下的最大等待时间,对于低频传感器通常设置为帧间隔时间的一半,
* 超过该时间数据还没到来,就认为本次打包该传感器数据异常,unit:s */
double time_wait_max_lidar = 0.10;

bool TryToFetchOneLidarFrame(MeasureGroup &meas)
{
#ifdef ENABLE_SENSOR_FAULT
    /* 如果lidar帧的时间戳晚于上一次包结束时间,说明lidar数据延迟太大,导致上一次打包时判定lidar为故障,
    * 因此需要丢弃过时的lidar帧 */
    while ((!lidar_buffer.empty()) && (time_end_buffer.front()<=last_time_packed)) 
    {
        lidar_buffer.pop_front();
        time_buffer.pop_front();
        time_end_buffer.pop_front();
    }
#endif   
    if (lidar_buffer.empty()) {
        return false;
    }    
    
    meas.lidar = lidar_buffer.front();
    meas.lidar_beg_time = time_buffer.front();

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    time_end_buffer.pop_front();

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
        // scan_num ++;
        lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
        // lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        lidar_mean_scantime = 0.1 ;
    }
    return true;

}

bool flag_roswarn_lidar = false;
//把当前要处理的LIDAR和IMU数据打包到meas
bool sync_packages(MeasureGroup &meas)
{
    // cout<<"1.1"<<endl;  
    if(!first_lidar)
    {
        return false;
    }
    
#ifdef ENABLE_SENSOR_FAULT
    /* 初始化last_time_packed为当前时间 */
    if (!initialized_last_time_packed) {
        last_time_packed = ros::Time::now().toSec();
        meas.package_end_time = last_time_packed;
        initialized_last_time_packed = true;
        return false;
    }
#endif
    // if (lidar_buffer.empty() || imu_buffer.empty()) {
    //     return false;
    // }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        if(TryToFetchOneLidarFrame(meas))
        {
            meas.lidar_end_time = lidar_end_time;
            meas.lidar_vaild = true;   
            lidar_pushed = true;  
            package_end_time_last = meas.package_end_time;
            meas.package_end_time = lidar_end_time; 

            if(flag_roswarn_lidar == true)
            {
                ROS_INFO("Lidar Data Recovery!");
                flag_roswarn_lidar = false;
            }   
        }
        else
        {
#ifdef ENABLE_SENSOR_FAULT
            /* 计时,距离上次打包成功后,一定时间内都没有数据,代表lidar故障 */
            if ((ros::Time::now().toSec() - last_time_packed) > (time_interval_packed + time_wait_max_lidar)) 
                {
                    // printf("  Error: SyncPackages: Lidar is malfunctioning!\n");
                    // cout<<"delta time:"<<ros::Time::now().toSec() - last_time_packed<<endl;
                    if(flag_roswarn_lidar == false && imu_vaild)
                    {
                        ROS_ERROR("Lidar Data Error !");
                        flag_roswarn_lidar = true;
                    }

                    lidar_pushed = true;
                    //雷达数据指向空指针
                    
                    PointCloudXYZI::Ptr  point_cloud_tmp(new PointCloudXYZI());
                    meas.lidar = point_cloud_tmp;
                    meas.lidar_vaild = false;
                    //测试一下imu递推是不是这个的问题
                    meas.lidar_beg_time = meas.package_end_time;
                    meas.lidar_end_time = meas.package_end_time + time_interval_packed;
                    /* 如果没有lidar数据,每次打包结束时间就是上一次加上time_interval_packed */
                    package_end_time_last = meas.package_end_time;
                    meas.package_end_time += time_interval_packed;
                }
#endif            
        }
    }
    // cout<<"1.1"<<endl;   
    /* 如果lidar还没取出(或判断为故障) */
    if (!lidar_pushed)
    return false;

    // if (last_timestamp_imu < lidar_end_time)
    // {
    //     return false;
    // }  

    if(imu_vaild)
    {
        //判断imu异常(lidar打包后0.5s都没有接收到imu数据)
        if((ros::Time::now().toSec() - meas.package_end_time)>0.5)
        {
            // cout<<"dt:"<<ros::Time::now().toSec()-meas.package_end_time<<endl;
            ROS_ERROR("IMU data error!!!");
            imu_vaild = false;
            lidar_pushed = false; //imu异常需要把lidar pop出去，否则会一直卡在这里
            if(imu_buffer.empty())
                imu_fault_time = meas.package_end_time;
            else
                imu_fault_time = imu_buffer.back()->header.stamp.toSec();
            // flg_exit = true;
            return false;
        }
    }
    else
    {
        lidar_pushed = false;
        return false;
    }


    //确保具备提取imu数据的条件，即最后一个imu时间大于包的结束时间
    if (last_timestamp_imu <= meas.package_end_time)
        return false;

    //确保具备提取leg数据的条件，即最后一个leg时间大于包的结束时间
    // if(useleg)
    // {
    //     if (last_timestamp_leg <= meas.package_end_time)
    //     return false;
    // }
    // cout<<"1.2"<<endl;  
    /*** push imu data, and pop from imu buffer ***/
    if(imu_buffer.empty()) 
        return false;

    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    // while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    while ((!imu_buffer.empty()) && (imu_time <= meas.package_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        // if(meas.package_end_time - imu_time > 0.2)
        // {
        //     imu_buffer.pop_front();
        //     continue;
        // }           
        // if(imu_time > lidar_end_time) break;
        if(imu_time > meas.package_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }
    // cout<<"imu_vaild:"<<!meas.imu.empty()<<endl;

    /*** push leg data, and pop from leg buffer ***/
    if(!leg_buffer.empty() && useleg)
    {
        double leg_time = leg_buffer.front()->header.stamp.toSec();
        meas.leg.clear();
        while ((!leg_buffer.empty()) && (leg_time <= meas.package_end_time))
        {
            leg_time = leg_buffer.front()->header.stamp.toSec();
            if(leg_time > meas.package_end_time) break;
            meas.leg.push_back(leg_buffer.front());
            leg_buffer.pop_front();
        }
    }
    // cout<<"1.3"<<endl;  
    //判断leg数据是否有效
    if(!meas.leg.empty())
    {
        meas.leg_vaild = true;
    }
    else
    {
        meas.leg_vaild = false;
    }

    // cout<<"leg_vaild:"<<meas.leg_vaild<<endl;

    lidar_pushed = false;

    last_time_packed = meas.package_end_time;
    // cout<<"1.4"<<endl;
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
    // V3D p_global(state_point.rot.matrix() * p_body + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}


void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I.matrix()*p_body_lidar + state_point.offset_T_L_I);
    // V3D p_body_imu = p_body_lidar;

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
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
        // publish_count -= PUBFRAME_PERIOD;
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
    // publish_count -= PUBFRAME_PERIOD;
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
    //转换到机器人中心坐标系下
    // Eigen::Vector3d tL_R,tW_R;
    // tL_R<<-0.12,0,-0.0162;//-0.353,0,-0.174
    // tW_R = state_point.rot *  tL_R + state_point.pos;
    // out.pose.position.x = tW_R[0];
    // out.pose.position.y = tW_R[1];
    // out.pose.position.z = tW_R[2];
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
        //队列长度超过100，移除最早的path
        if(path.poses.size()>100)
        {
            path.poses.erase(path.poses.begin());
        }
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

/**
 * Eigen格式的位姿变换
 */
Eigen::Affine3f pclPointToAffine3f(PointTypePose thisPoint)
{
    return pcl::getTransformation(thisPoint.x, thisPoint.y, thisPoint.z, thisPoint.roll, thisPoint.pitch, thisPoint.yaw);
}

/**
 * Eigen格式的位姿变换
 */
Eigen::Affine3f trans2Affine3f(float transformIn[])
{
    return pcl::getTransformation(transformIn[3], transformIn[4], transformIn[5], transformIn[0], transformIn[1], transformIn[2]);
}

/**
 * 位姿格式变换
 */
gtsam::Pose3 pclPointTogtsamPose3(PointTypePose thisPoint)
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(double(thisPoint.roll), double(thisPoint.pitch), double(thisPoint.yaw)),
                        gtsam::Point3(double(thisPoint.x), double(thisPoint.y), double(thisPoint.z)));
}

/**
 * 位姿格式变换
 */
gtsam::Pose3 trans2gtsamPose(float transformIn[])
{
    return gtsam::Pose3(gtsam::Rot3::RzRyRx(transformIn[0], transformIn[1], transformIn[2]),
                        gtsam::Point3(transformIn[3], transformIn[4], transformIn[5]));
}

//  eulerAngle 2 Quaterniond
Eigen::Quaterniond  EulerToQuat(float roll_, float pitch_, float yaw_)
{
    Eigen::Quaterniond q ;            //   四元数 q 和 -q 是相等的
    Eigen::AngleAxisd roll(double(roll_), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch(double(pitch_), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw(double(yaw_), Eigen::Vector3d::UnitZ());
    q = yaw * pitch * roll ;
    q.normalize();
    return q ;
}

/**
 * 两点之间距离
 */
float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z));
}

// void ReadRtkParams()
// {
//   fstream infile_Rtk(string(loadrtkpath+"Rtk.txt"));
//   string TextLine;
//   vector<string> Re;
//   getline(infile_Rtk,TextLine);
//   RtkAngle=atof(TextLine.c_str());
//   getline(infile_Rtk,TextLine);
//   RtkPos[0]=atof(TextLine.c_str());
//   getline(infile_Rtk,TextLine);
//   RtkPos[1]=atof(TextLine.c_str());
//   getline(infile_Rtk,TextLine);
//   RtkPos[2]=atof(TextLine.c_str());
//   getline(infile_Rtk,TextLine);
//   Re=Split(TextLine,", ");
//   RtkConvert(0,0)=atof(Re[0].c_str());
//   RtkConvert(0,1)=atof(Re[1].c_str());
//   RtkConvert(0,2)=atof(Re[2].c_str());
//   RtkConvert(0,3)=atof(Re[3].c_str());
//   getline(infile_Rtk,TextLine);
//   Re=Split(TextLine,", ");
//   RtkConvert(1,0)=atof(Re[0].c_str());
//   RtkConvert(1,1)=atof(Re[1].c_str());
//   RtkConvert(1,2)=atof(Re[2].c_str());
//   RtkConvert(1,3)=atof(Re[3].c_str());
//   getline(infile_Rtk,TextLine);
//   Re=Split(TextLine,", ");
//   RtkConvert(2,0)=atof(Re[0].c_str());
//   RtkConvert(2,1)=atof(Re[1].c_str());
//   RtkConvert(2,2)=atof(Re[2].c_str());
//   RtkConvert(2,3)=atof(Re[3].c_str());
//   getline(infile_Rtk,TextLine);
//   Re=Split(TextLine,", ");
//   RtkConvert(3,0)=atof(Re[0].c_str());
//   RtkConvert(3,1)=atof(Re[1].c_str());
//   RtkConvert(3,2)=atof(Re[2].c_str());
//   RtkConvert(3,3)=atof(Re[3].c_str());

//   double Lat=msg_in->lat, Lon=msg_in->lon, Hgt=msg_in->hgt;
//   int zone;
//   bool northp;
//   double x, y, z;
//   GeographicLib::UTMUPS::Forward(Lat, Lon, zone, northp, x, y);
//   z=Hgt=msg_in->hgt;

//   x-=RtkPos[0];   
//   y-=RtkPos[1];
//   z-=RtkPos[2];

//   Eigen::Vector3f xyz_rtk_ori(x,y,z), xyz_odom(state_point.pos[0],state_point.pos[1],state_point.pos[2]);
//   Eigen::Matrix3f Rot=RtkConvert.block<3,3>(0,0);
//   Eigen::Vector3f Pos=RtkConvert.block<3,1>(0,3);
//   xyz_rtk_ori=Rot*xyz_rtk_ori+Pos;
//   Eigen::Vector3d xyz_rtk(xyz_rtk_ori[0],xyz_rtk_ori[1],xyz_rtk_ori[2]);

//   gnss_heading=gnss_heading+RtkAngle;
//   if(gnss_heading<-3.1415926)
//     gnss_heading+=2*3.1415926;
//   if(gnss_heading>3.1415926)
//     gnss_heading-=2*3.1415926;
// }



void gnss_cbk(const na_localization::rtk_pos_raw::ConstPtr& msg_in) //todo yaml更改 usertk
{
    //判断是否使用rtk融合
    if (usertk==false)
        return;
    //判断gps是否有效

    // if (msg_in->pos_type!=16)  //todo =50 rtk 模式  //todo =16 gnss单点定位
    //     return;

    // if (msg_in->hgt_std_dev > 0.05) //todo 同理 这个判定是rtk用的判定
    //     return;

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

    //通过gps_qual给出协方差
    double posecov;
    posecov=0.05*0.05;

    gnss_data.pose_cov[0] = posecov;
    gnss_data.pose_cov[1] = posecov;
    gnss_data.pose_cov[2] = 2.0*posecov;

    mtx_buffer.unlock();
    double Lat = msg_in->lat, Lon = msg_in->lon;
    int zone;
    bool northp;
    double x_,y_,z_;
    GeographicLib::UTMUPS::Forward(Lat, Lon, zone, northp, x_, y_);
    z_ = msg_in->hgt;
    x_ -= 669369.25000;
    y_ -= 3528527.25000;
    z_ -= 17.28488;

    M3D rotation_tmp;
    V3D drift_tmp,position_tmp;
    drift_tmp << -196.486,153.127,3.44237;
    rotation_tmp << 0.0972805, -0.995257, -0.000182575,
                    0.99526, 0.0972791, -0.000187838,
                    0.00020471, -0.000163439, 1;
    position_tmp << x_,y_,z_;
    position_tmp = rotation_tmp * position_tmp + drift_tmp;
    x_ = position_tmp[0];
    y_ = position_tmp[1];
    z_ = position_tmp[2];
    // gnss_data.UpdateXYZ(msg_in->lat, msg_in->lon, msg_in->hgt) ;             //  WGS84 -> ENU  ???  调试结果好像是 NED 北东地
    nav_msgs::Odometry gnss_data_enu ;
    // add new message to buffer:
    gnss_data_enu.header.stamp = ros::Time().fromSec(gnss_data.time);
    gnss_data_enu.pose.pose.position.x =  x_ ;  //gnss_data.local_E ;   北
    gnss_data_enu.pose.pose.position.y =  y_ ;  //gnss_data.local_N;    东
    gnss_data_enu.pose.pose.position.z =  z_;  //  地

    gnss_data_enu.pose.covariance[0] = posecov ;
    gnss_data_enu.pose.covariance[7] = posecov ;
    gnss_data_enu.pose.covariance[14] = 2.0*posecov ;

    gnss_buffer.push_back(gnss_data_enu);
    if(gnss_buffer.size()>50)
    {
        gnss_buffer.pop_front();
        // ROS_WARN("gnss buffer is too large!");
    }
    // visial gnss path in rviz:
    msg_gnss_pose.header.frame_id = "camera_init";
    msg_gnss_pose.header.stamp = ros::Time().fromSec(gnss_data.time);
    // Eigen::Vector3d gnss_pose_ (gnss_data.local_E, gnss_data.local_N, - gnss_data.local_U); 
    // Eigen::Vector3d gnss_pose_ (gnss_data.local_N, gnss_data.local_E, - gnss_data.local_U); 
    Eigen::Matrix4d gnss_pose = Eigen::Matrix4d::Identity();

    gnss_pose(0,3) = x_ ;
    gnss_pose(1,3) = y_ ;
    gnss_pose(2,3) = z_ ; 

    // Eigen::Isometry3d gnss_to_lidar(Gnss_R_wrt_Lidar) ;
    // gnss_to_lidar.pretranslate(Gnss_T_wrt_Lidar);
    // gnss_pose  =  gnss_to_lidar  *  gnss_pose ;                    //  gnss 转到 lidar 系下

    msg_gnss_pose.pose.position.x = gnss_pose(0,3) ;  
    msg_gnss_pose.pose.position.y = gnss_pose(1,3) ;
    msg_gnss_pose.pose.position.z = gnss_pose(2,3) ;
    
    //队列长度超过100，移除最早的path
    if(gps_path.poses.size()>100)
    {
        gps_path.poses.erase(gps_path.poses.begin());
    }
    gps_path.poses.push_back(msg_gnss_pose); 

    if(need_relocal)
    {
        if(flag_rtkpos == false)
        {
            flag_rtkpos = true;
        }
    } 

}

// void gnss_cbk(const na_localization::rtk_pos_raw::ConstPtr& msg_in)
// {
//     //判断是否使用rtk融合
//     // if (usertk==false)
//     //     return;
//     //判断gps是否有效
//     if (msg_in->pos_type!=50)
//         return;
//     if (msg_in->hgt_std_dev > 0.05)
//         return;
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

//     gnss_data.pose_cov[0] = posecov;
//     gnss_data.pose_cov[1] = posecov;
//     gnss_data.pose_cov[2] = 2.0*posecov;

//     mtx_buffer.unlock();
//     gnss_data.UpdateXYZ(msg_in->lat, msg_in->lon, msg_in->hgt) ;             //  WGS84 -> ENU  ???  调试结果好像是 NED 北东地
//     nav_msgs::Odometry gnss_data_enu ;
//     // add new message to buffer:
//     gnss_data_enu.header.stamp = ros::Time().fromSec(gnss_data.time);
//     gnss_data_enu.pose.pose.position.x =  gnss_data.local_E ;  //gnss_data.local_E ;   北
//     gnss_data_enu.pose.pose.position.y =  gnss_data.local_N ;  //gnss_data.local_N;    东
//     gnss_data_enu.pose.pose.position.z =  gnss_data.local_U;  //  地

//     gnss_data_enu.pose.covariance[0] = gnss_data.pose_cov[0] ;
//     gnss_data_enu.pose.covariance[7] = gnss_data.pose_cov[1] ;
//     gnss_data_enu.pose.covariance[14] = gnss_data.pose_cov[2] ;

//     gnss_buffer.push_back(gnss_data_enu);

//     // visial gnss path in rviz:
//     msg_gnss_pose.header.frame_id = "camera_init";
//     msg_gnss_pose.header.stamp = ros::Time().fromSec(gnss_data.time);
//     // Eigen::Vector3d gnss_pose_ (gnss_data.local_E, gnss_data.local_N, - gnss_data.local_U); 
//     // Eigen::Vector3d gnss_pose_ (gnss_data.local_N, gnss_data.local_E, - gnss_data.local_U); 
//     Eigen::Matrix4d gnss_pose = Eigen::Matrix4d::Identity();

//     gnss_pose(0,3) = gnss_data.local_E ;
//     gnss_pose(1,3) = gnss_data.local_N ;
//     gnss_pose(2,3) =gnss_data.local_U ;

//     // Eigen::Isometry3d gnss_to_lidar(Gnss_R_wrt_Lidar) ;
//     // gnss_to_lidar.pretranslate(Gnss_T_wrt_Lidar);
//     // gnss_pose  =  gnss_to_lidar  *  gnss_pose ;                    //  gnss 转到 lidar 系下

//     msg_gnss_pose.pose.position.x = gnss_pose(0,3) ;  
//     msg_gnss_pose.pose.position.y = gnss_pose(1,3) ;
//     msg_gnss_pose.pose.position.z = gnss_pose(2,3) ;

//     gps_path.poses.push_back(msg_gnss_pose); 

//     if(need_relocal)
//     {
//         if(flag_rtkpos == false)
//         {
//             flag_rtkpos = true;
//         }
//     } 

// }

void gnss_heading_cbk(const na_localization::rtk_heading_raw::ConstPtr& msg_in)
{
    //如果不是固定解或不使用rtk的航向,return;
    if(msg_in->pos_type != 50)
    {
        return ;
    }
    
    // rtk_heading = (180.0 - msg_in->heading) * M_PI / 180.0;
    // rtk_heading = - msg_in->heading * M_PI / 180.0;
    rtk_heading = 360.0 - msg_in->heading + 87.0;
    // rtk_heading = msg_in->heading;

    if(rtk_heading<0)
    {
        rtk_heading += 360.0;
    }
    else if(rtk_heading>360.0)
    {
        rtk_heading -= 360;
    }

    rtk_heading = rtk_heading * M_PI / 180.0;

    gnss_heading_buffer.push_back(rtk_heading);
    heading_time_buffer.push_back(msg_in->header.stamp.toSec());
    if(gnss_heading_buffer.size()>50)
    {
        gnss_heading_buffer.pop_front();
        heading_time_buffer.pop_front();
        // ROS_WARN("gnss_heading_buffer is too large!");
    }
    if(need_relocal)
    {
        if(flag_rtkheading == false)
        {
            flag_rtkheading = true;
        }
    }
}

void ManualPos_cbk(const geometry_msgs::PoseStamped::ConstPtr &msg_in)
{
    // if (Reposition_type!=2)
    //     return; 

    if (flag_manualpos==false)
    {
       manualpos[0]=msg_in->pose.position.x;
       manualpos[1]=msg_in->pose.position.y;
       manualpos[2]=msg_in->pose.position.z; 

       ext_q.x()=msg_in->pose.orientation.x;
       ext_q.y()=msg_in->pose.orientation.y;
       ext_q.z()=msg_in->pose.orientation.z;
       ext_q.w()=msg_in->pose.orientation.w;

       flag_manualpos=true;


       cout<<"Receive External Position Successful!"<<endl;
       cout<<manualpos[0]<<" "<<manualpos[1]<<" "<<manualpos[2]<<endl;
    } 

}

void manualpos_reposition()
{
    //将标志位置false;
    flag_manualpos = false;

    double min_score=100.0;
    Eigen::Matrix4f transformation=Eigen::Matrix4f::Identity();

    pcl::PointCloud<PointType>::Ptr mapreposition(new pcl::PointCloud<PointType>());
    PointType pos_repos;

    pos_repos.x=manualpos[0];
    pos_repos.y=manualpos[1];
    pos_repos.z=manualpos[2];
    //初始匹配的地图，半径设置为80m
    for(int i=0;i<(int)pointcloudmap->size();i++)
    {
        if (pointDistance(pointcloudmap->points[i], pos_repos) > 80.0)
        continue;
                        
        mapreposition->push_back(pointcloudmap->points[i]);
    }

    if(mapreposition->points.size() < 100)
    {
        ROS_WARN("Invaild Position");
        return;
    }


    Eigen::Matrix3d rotation_matrix;
    rotation_matrix=ext_q.toRotationMatrix();
    // rotation_matrix=Angle2Matrix(yaw0);`
    Eigen::Matrix4f preTF=Matrix4f::Identity();

    preTF.block<3,3>(0,0)=rotation_matrix.cast<float>();
    preTF(0,3)=(float)manualpos[0];
    preTF(1,3)=(float)manualpos[1];
    preTF(2,3)=(float)manualpos[2];

    // pcl::PointCloud<PointCloudXYZI> pl_orig;
    PointCloudXYZI::Ptr pl_orig (new PointCloudXYZI);//声明源点云
    pcl::PointCloud<PointType>::Ptr ds_pl_orig (new pcl::PointCloud<PointType>());//声明源点云
    //如果没有雷达点云数据，直接return
    if(!first_lidar) 
    {
        cout<<"lidar no points"<<endl;
        return;
    }

    pl_orig = Measures.lidar;
    //cout<<"point cloud size:"<<pl_orig->points.size()<<endl;
    pcl::VoxelGrid<PointType> sor;
    sor.setInputCloud(pl_orig);
    sor.setLeafSize(0.1f,0.1f,0.1f);
    sor.filter(*ds_pl_orig);

    // flag_lidar=true;
    // cout<<"lidar ready!"<<endl;

    // cout<<preTF<<endl;
    //调用icp匹配，得到精位姿
    
    pcl::IterativeClosestPoint<PointType,PointType> icp1;
    icp1.setInputSource(ds_pl_orig);
    icp1.setInputTarget(mapreposition);
    icp1.setMaxCorrespondenceDistance (150);
    icp1.setMaximumIterations (20);
    icp1.setTransformationEpsilon (1e-6);
    icp1.setEuclideanFitnessEpsilon (1e-6);
    icp1.setRANSACIterations(0);
    pcl::PointCloud<PointType> Final;
    icp1.align(Final,preTF);


    //todo NDT
    // pcl::NormalDistributionsTransform<PointType,PointType> ndt;  //todo NDT
    // ndt.setInputSource(ds_pl_orig);
    // ndt.setInputTarget(mapreposition);
    // ndt.setStepSize(0.1);
    // ndt.setResolution(0.5);
    // ndt.setMaximumIterations(30);
    // ndt.setTransformationEpsilon(0.01);
    // pcl::PointCloud<PointType> Final;
    // ndt.align(Final,preTF);
    // transformation = ndt.getFinalTransformation();

    transformation = icp1.getFinalTransformation();
    // cout<<"score:"<<min_score<<endl;
    cout<<"score:"<<icp1.getFitnessScore()<<endl;
    // cout<<"score:"<<ndt.getFitnessScore()<<endl;
    //ICP匹配度太差，重定位失败
    if(icp1.getFitnessScore() > 1.0)
    {
        ROS_WARN("ICP Score is too big!");
        return;
    }
    // if(ndt.getFitnessScore() > 1.0)
    // {
    //     ROS_WARN("NDT Score is too big!");
    //     return;
    // }
    cout<<transformation<<endl;

    //规范一下旋转矩阵，否则Sophus库会出问题
    Eigen::Matrix3d tmp_matrix =transformation.block<3,3>(0,0).cast<double>();
    Eigen::Quaterniond tmp_quat= Eigen::Quaterniond(tmp_matrix);
    tmp_matrix = tmp_quat.normalized().toRotationMatrix();
    // Eigen::Matrix3f tmp_matrix =transformation.block<3,3>(0,0);
    // cout<<tmp_matrix*tmp_matrix.transpose()<<endl;
    // state_point = kf.get_x(); 
    state_point = state_point_last;
    state_point.rot=Sophus::SO3d(tmp_matrix);
    state_point.pos=transformation.block<3,1>(0,3).cast<double>();
    //cout<<state_point.rot<<endl;
    kf.change_x(state_point);
    cout<<"重定位完成！"<<endl;
    // flag_reposition=true;
    need_relocal = false;
}

void rtk_reloc()
{
    //将标志位置false;
    flag_rtkpos = false; //todo 仅使用rtk三维坐标信息 不使用航向信息

    cout << 1111 << endl;

    nav_msgs::Odometry gnss_tmp = gnss_buffer.back(); //Odometry数据类型  把GNNS数据存储内容放进去
    PointType pos_repos; //pointXYZ类型
    pos_repos.x = gnss_tmp.pose.pose.position.x; //point类型 xyz 等于 gnss获取的xyz
    pos_repos.y = gnss_tmp.pose.pose.position.y;
    pos_repos.z = gnss_tmp.pose.pose.position.z;

    if(first_lidar == false)
            return;

    reloc_plugin_ptr_->readLidar(trans_cloud);  //!


    cerr << "local relco call !!!" << endl;

    double search_radius = 10.0;
    bool reloSuccess_ = false;
    utils::Pose init_pose(pos_repos.x, pos_repos.y, 3.0, 0, 0, 0);

    reloSuccess_ = reloc_plugin_ptr_->localRelocByBfs(init_pose, search_radius);

    if(reloSuccess_)
        {
            utils::Pose reloc_pose = reloc_plugin_ptr_->getRelocPose();
            state_point = kf.get_x(); 
            state_point.rot=Sophus::SO3d(reloc_pose.q_);
            state_point.pos=reloc_pose.t_;

            kf.change_x(state_point);
            need_relocal = false;

            cout<<"局部重定位成功"<<endl;
        }
        else
        {
            cout<<"局部重定位失败"<<endl;
        }
}




// void rtk_reposition() //todo 12.5 调整 从RTK里面读取的坐标 偏置在 x y z上面加 1.2——2m 测试是否还能重定位成功
//                       //todo roslaunch之后 先跑包 （测试RTK重定位） 首先 --clock之后
//                       //todo 在跑包的终端 空格暂停 给重定位运算时间
//                       //todo 直到显示重定位成功 再次空格取消读包暂停 
//                       //! 不暂停跑包读取重定位 极大概率会飘
//                       //todo 在 x y z 大概取+3.5m基本上就飘了
//                       //todo z的方向上能承受+4的误差
//                       //todo 2024.1.4 use change 
// {
//     //将标志位置false;
//     flag_rtkpos = false;
//     flag_rtkheading = false;

//     pcl::PointCloud<PointType>::Ptr mapreposition(new pcl::PointCloud<PointType>()); //点云类型智能指针 
//     PointType pos_repos; //pointXYZ类型
//     nav_msgs::Odometry gnss_tmp = gnss_buffer.back(); //Odometry数据类型  把GNNS数据存储内容放进去
//     pos_repos.x = gnss_tmp.pose.pose.position.x; //point类型 xyz 等于 gnss获取的xyz
//     pos_repos.y = gnss_tmp.pose.pose.position.y;
//     pos_repos.z = gnss_tmp.pose.pose.position.z;
    
//     //初始匹配的地图，半径设置为80m
//     for(int i=0;i<(int)pointcloudmap->size();i++)
//     {
//         if (pointDistance(pointcloudmap->points[i], pos_repos) > 50.0)
//         continue;

//         mapreposition->push_back(pointcloudmap->points[i]);
//     }
//     //初始位置错误导致点云数量太少
//     if(mapreposition->points.size() < 100)
//     {
//         ROS_WARN("Invaild Position");
//         return;
//     }
//     PointCloudXYZI::Ptr pl_orig (new PointCloudXYZI);//声明源点云
//     pcl::PointCloud<PointType>::Ptr ds_pl_orig (new pcl::PointCloud<PointType>());//声明源点云
//     //如果没有雷达点云数据，直接return
//     if(!first_lidar) 
//     {
//         cout<<"lidar no points"<<endl;
//         return;
//     }

//     pl_orig = Measures.lidar;
//     if(pl_orig->points.size()<5)
//     return;
//     pcl::VoxelGrid<PointType> sor;
//     sor.setInputCloud(pl_orig);
//     // sor.setLeafSize(0.1f,0.1f,0.1f);
//     sor.setLeafSize(0.4f,0.4f,0.4f);
//     sor.filter(*ds_pl_orig);

//     //角度遍历
//     double min_score=100.0;
//     Eigen::Matrix4f transformation=Eigen::Matrix4f::Identity();

//     for(int Id=1;Id<=18;Id++)
//     {
//         double yaw0=(Id-1)*20*M_PI/180.0;
//         // Eigen::Matrix3d rotation_matrix;
//         // Eigen::Matrix3d Angle2Matrix(double yaw);
//         // rotation_matrix=Eigen::Angle2Matrix(yaw0);

//         Eigen::AngleAxisd rot_z_btol(yaw0, Eigen::Vector3d::UnitZ());//旋转向量 （旋转角 旋转轴 只用z ）
//         Eigen::Matrix3d rotation_matrix = rot_z_btol.matrix();//矩阵化

//         Eigen::Matrix4f preTF=Matrix4f::Identity();

//         preTF.block<3,3>(0,0)=rotation_matrix.cast<float>();
//         preTF(0,3)=(float)pos_repos.x+3.3;
//         preTF(1,3)=(float)pos_repos.y+3.3;
//         preTF(2,3)=(float)pos_repos.z+3.3;

//         pcl::IterativeClosestPoint<PointType,PointType> icp1;       //todo ICP
//         icp1.setInputSource(ds_pl_orig);
//         icp1.setInputTarget(mapreposition);
//         icp1.setMaxCorrespondenceDistance (150);
//         icp1.setMaximumIterations (20);
//         icp1.setTransformationEpsilon (1e-6);
//         icp1.setEuclideanFitnessEpsilon (1e-6);
//         icp1.setRANSACIterations(0);
//         pcl::PointCloud<PointType> Final;
//         icp1.align(Final,preTF); 


//         // pcl::NormalDistributionsTransform<PointType,PointType> ndt;  //todo NDT
//         // // pcl::PointCloud<PointType>::Ptr aligned_cloud(new pcl::PointCloud<PointType>());
//         // ndt.setInputSource(ds_pl_orig);
//         // ndt.setInputTarget(mapreposition);
//         // ndt.setStepSize(0.1);
//         // ndt.setResolution(1.0);
//         // ndt.setMaximumIterations(30);
//         // ndt.setTransformationEpsilon(0.01);
//         // // ndt.align(*aligned_cloud);
//         // pcl::PointCloud<PointType> Final;
//         // ndt.align(Final,preTF); 

//         // if(ndt.getFitnessScore()<min_score)
//         // {
//         //     transformation = ndt.getFinalTransformation();
//         //     min_score=ndt.getFitnessScore();
//         // }

//         if(icp1.getFitnessScore()<min_score)
//         {
//             transformation = icp1.getFinalTransformation();
//             min_score=icp1.getFitnessScore();
//         }
//         cout<<"yaw:"<<yaw0*180/M_PI<<endl;
//         cout<<"score:"<<icp1.getFitnessScore()<<endl;
//         // cout<<"score:"<<ndt.getFitnessScore()<<endl;
//     }

//     cout<<"score:"<<min_score<<endl;
//     cout<<transformation<<endl;

//     Eigen::Matrix3d tmp_matrix =transformation.block<3,3>(0,0).cast<double>();
//     Eigen::Quaterniond tmp_quat= Eigen::Quaterniond(tmp_matrix);
//     tmp_matrix = tmp_quat.normalized().toRotationMatrix();
//     state_point = state_point_last;

//     state_point = kf.get_x(); 
//     state_point.rot=Sophus::SO3d(tmp_matrix);
//     // state_point.rot=transformation.block<3,3>(0,0).cast<double>();
//     state_point.pos=transformation.block<3,1>(0,3).cast<double>(); //todo 
//     //     state_point.rot=Sophus::SO3d(tmp_matrix);

//     // cout<<state_point.rot<<endl;
//     kf.change_x(state_point);
//     cout<<"重定位完成！"<<endl;
//     need_relocal = false;
// }





void rtk_reposition()    //todo 2024.1.4 use before
{
    //将标志位置false;
    flag_rtkpos = false;
    flag_rtkheading = false;

    pcl::PointCloud<PointType>::Ptr mapreposition(new pcl::PointCloud<PointType>()); //点云类型智能指针 
    PointType pos_repos; //pointXYZ类型
    nav_msgs::Odometry gnss_tmp = gnss_buffer.back(); //Odometry数据类型  把GNNS数据存储内容放进去
    pos_repos.x = gnss_tmp.pose.pose.position.x; //point类型 xyz 等于 gnss获取的xyz
    pos_repos.y = gnss_tmp.pose.pose.position.y;
    pos_repos.z = gnss_tmp.pose.pose.position.z;
    
    //初始匹配的地图，半径设置为80m
    for(int i=0;i<(int)pointcloudmap->size();i++)
    {
        if (pointDistance(pointcloudmap->points[i], pos_repos) > 80.0)
        continue;
                        
        mapreposition->push_back(pointcloudmap->points[i]);
    }

    //初始位置错误导致点云数量太少
    if(mapreposition->points.size() < 100)
    {
        ROS_WARN("Invaild Position");
        return;
    }

    Eigen::AngleAxisd rot_z_btol(rtk_heading, Eigen::Vector3d::UnitZ());//旋转向量 （旋转角 旋转轴 只用z ）
    Eigen::Matrix3d rotation_matrix = rot_z_btol.matrix();//矩阵化
    Eigen::Matrix4f preTF=Matrix4f::Identity(); //初始化四维单位矩阵

    preTF.block<3,3>(0,0)=rotation_matrix.cast<float>();
    preTF(0,3)=(float)pos_repos.x; //第一列第四个
    preTF(1,3)=(float)pos_repos.y;
    preTF(2,3)=(float)pos_repos.z;

    PointCloudXYZI::Ptr pl_orig (new PointCloudXYZI);//声明源点云
    pcl::PointCloud<PointType>::Ptr ds_pl_orig (new pcl::PointCloud<PointType>());//声明源点云
    //如果没有雷达点云数据，直接return
    if(!first_lidar) 
    {
        cout<<"lidar no points"<<endl;
        return;
    }

    pl_orig = Measures.lidar;
    if(pl_orig->points.size()<5)
    return;
    pcl::VoxelGrid<PointType> sor;
    sor.setInputCloud(pl_orig);
    sor.setLeafSize(0.1f,0.1f,0.1f);
    sor.filter(*ds_pl_orig);

    //调用icp匹配，得到精位姿
    pcl::IterativeClosestPoint<PointType,PointType> icp1;
    icp1.setInputSource(ds_pl_orig);
    icp1.setInputTarget(mapreposition);
    icp1.setMaxCorrespondenceDistance (150);
    icp1.setMaximumIterations (20);
    icp1.setTransformationEpsilon (1e-6);
    icp1.setEuclideanFitnessEpsilon (1e-6);
    icp1.setRANSACIterations(0);
    pcl::PointCloud<PointType> Final;
    icp1.align(Final,preTF);

    cout<<"score:"<<icp1.getFitnessScore()<<endl;
    //ICP匹配度太差，重定位失败
    if(icp1.getFitnessScore() > 1.0)
    {
        ROS_WARN("ICP Score is too big!");
        return;
    }

    Eigen::Matrix4f transformation=Eigen::Matrix4f::Identity();
    transformation = icp1.getFinalTransformation();

    Eigen::Matrix3d tmp_matrix =transformation.block<3,3>(0,0).cast<double>();
    Eigen::Quaterniond tmp_quat= Eigen::Quaterniond(tmp_matrix);
    tmp_matrix = tmp_quat.normalized().toRotationMatrix();
    state_point = state_point_last;
    state_point.rot=Sophus::SO3d(tmp_matrix);
    state_point.pos=transformation.block<3,1>(0,3).cast<double>();

    kf.change_x(state_point);
    cout<<"重定位完成！"<<endl;
    need_relocal = false;
}

//重构ikdtree线程
// void ReikdtreeThread()
// {
//     // cout<<"******************"<<endl;
//     ros::Rate rate1(1);
//     while (ros::ok())
//     {
//         if (flg_exit)
//             break;
//         if (flag_ikdtree_initial==true)
//         {

//             pos_now.x=state_point.pos[0];
//             pos_now.y=state_point.pos[1];
//             pos_now.z=state_point.pos[2];

//             if (pointDistance(pos_now, pos_last) > 30.0)
//             {
                    
//                 // pcl::PointCloud<PointType>::Ptr mapikdtree(new pcl::PointCloud<PointType>());
//                 mapikdtree->clear();

//                 for(int i=0;i<(int)pointcloudmap->size();i++)
//                 {
//                     if (pointDistance(pointcloudmap->points[i], pos_now) > 80.0)
//                     continue;
                        
//                     mapikdtree->push_back(pointcloudmap->points[i]);
//                 }
//                 // ikdtree.reconstruct(mapikdtree->points);
//                 pos_last=pos_now;

//                 mtx_reikd.lock();
//                 flag_reikdtree=true;
//                 mtx_reikd.unlock();
//                 cout<<"Rebuild IKdtree Successful! current pos is: x:"<<pos_now.x<<" y:"<<pos_now.y<<" z:"<<pos_now.z<<endl;
//                 // cout<<"Rebuild IKdtree Successful!"<<endl;
//             }
//         }
//         rate1.sleep();
//     }
    
// }



//重定位
void ReLocalization()
{
    //判断是否需要重定位
    if(!need_relocal) 
    {
        return;
    }

    //定期发布全局地图      
    if(map_count<100)
    {
        map_count++;
    }
    else
    {
        pubglobalmap.publish(globalmapmsg);
        map_count=0;
    }

    //选择重定位方式
    if(flag_rtkpos==true )//rtk重定位   && flag_rtkheading==true
    {
        // rtk_reposition(); //todo rtk重定位
        // return ;
        rtk_reloc();  //todo gnss 单点重定位
    }
    else if(flag_manualpos==true)//手动重定位 
    {
        manualpos_reposition();
        // return ;
    }
    

    //重定位完成
    if(!need_relocal)
    {
        //ikdtree初始化完成，说明本次重定位是定位丢失后的重定位，需要重置ikdtree和包围盒
        if(flag_ikdtree_initial)
        {
            MM.get_map(state_point.pos[0],state_point.pos[1]);
            // ikdtree.reconstruct(MM.pointcloud_output->points);

            kdtree1.setInputCloud(MM.pointcloud_output);
            kdtree2.setInputCloud(MM.pointcloud_output);

            // V3D pos_LiD = state_point.pos;  // W系下位置
            // for (int i = 0; i < 3; i++)
            // {
            //     LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            //     LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
            // }
        }
    }
    
}

void sensor_vaild_judge(Eigen::Vector3d &z_leg, Eigen::Vector3d &z_rtk)
{
    //判断leg的可用性
    // Eigen::Vector3d z_leg = Eigen::Vector3d::Zero();
    if(Measures.leg_vaild == true)
    {
        nav_msgs::Odometry::ConstPtr leg_back = Measures.leg.back();
        //z_leg只有前向和左向的速度
        z_leg(0)=leg_back->twist.twist.linear.x;
        z_leg(1)=leg_back->twist.twist.linear.y;
        // cout<<"velocity forward:"<<z_leg(0)<<" velocity left:"<<z_leg(1)<<endl;
    }
            
    //判断rtk的可用性
    rtk_vaild = false;
    // Eigen::Vector3d z_rtk = Eigen::Vector3d::Zero();
    while (!gnss_buffer.empty())
    {
        // 删除当前帧0.1s之前的rtk
        if (gnss_buffer.front().header.stamp.toSec() < Measures.package_end_time - 0.10)
        {
            gnss_buffer.pop_front();
        }
        // 超过当前帧0.05s之后，退出
        else if (gnss_buffer.front().header.stamp.toSec() > Measures.package_end_time)
        {
            break;
        }
        else
        {
            nav_msgs::Odometry thisGPS = gnss_buffer.front();
            gnss_buffer.pop_front();

            // V3D dp;
            // dp=state_point.rot.matrix()*rtk_T_wrt_Lidar;

            bool first_imu = true;
            double dt_rtk = 0.005;
            Sophus::SO3d rot_rtk = state_point_last.rot;
            V3D vel_rtk = state_point_last.vel;
            V3D angvel_avr,acc_avr;
            double imu_time_last = package_end_time_last;
            //丢掉上一个lidar时刻前的imu数据
            while(!imu_rtk_buffer.empty() && (package_end_time_last>0) && (imu_rtk_buffer.front()->header.stamp.toSec()<package_end_time_last))
            {
                imu_rtk_buffer.pop_front();
            }
            //更新姿态
            while(!imu_rtk_buffer.empty() && (package_end_time_last>0) && (imu_rtk_buffer.front()->header.stamp.toSec() <= thisGPS.header.stamp.toSec()))
            {
                //计算dt
                if(first_imu)
                {
                    dt_rtk = imu_rtk_buffer.front()->header.stamp.toSec() - package_end_time_last;
                    first_imu = false;
                    imu_time_last = imu_rtk_buffer.front()->header.stamp.toSec();
                }
                else
                {
                    dt_rtk = imu_rtk_buffer.front()->header.stamp.toSec() - imu_time_last;
                    imu_time_last = imu_rtk_buffer.front()->header.stamp.toSec();
                }

                
                angvel_avr << imu_rtk_buffer.front()->angular_velocity.x,
                              imu_rtk_buffer.front()->angular_velocity.y,
                              imu_rtk_buffer.front()->angular_velocity.z;

                acc_avr << imu_rtk_buffer.front()->linear_acceleration.x,
                           imu_rtk_buffer.front()->linear_acceleration.y,
                           imu_rtk_buffer.front()->linear_acceleration.z;

                rot_rtk = rot_rtk * Sophus::SO3d::exp(angvel_avr * dt_rtk);     
                acc_avr = rot_rtk * acc_avr + state_point_last.grav;
                vel_rtk += acc_avr * dt_rtk;

                imu_rtk_buffer.pop_front();           
            }

            V3D dp;
            dp = rot_rtk.matrix()*rtk_T_wrt_Lidar;
            //补偿杆臂
            z_rtk[0] = thisGPS.pose.pose.position.x-dp[0];
            z_rtk[1] = thisGPS.pose.pose.position.y-dp[1];
            z_rtk[2] = thisGPS.pose.pose.position.z-dp[2];

            //丢掉rtk时刻前的imu数据
            while(!imu_rtk_buffer.empty() && (package_end_time_last>0) && (imu_rtk_buffer.front()->header.stamp.toSec()<thisGPS.header.stamp.toSec()))
            {
                imu_rtk_buffer.pop_front();
            }
            //计算雷达结束时刻姿态和速度
            while(!imu_rtk_buffer.empty() && (package_end_time_last>0) && (imu_rtk_buffer.front()->header.stamp.toSec()<Measures.package_end_time))
            {
                dt_rtk = imu_rtk_buffer.front()->header.stamp.toSec() - imu_time_last;
                imu_time_last = imu_rtk_buffer.front()->header.stamp.toSec();

                angvel_avr << imu_rtk_buffer.front()->angular_velocity.x,
                              imu_rtk_buffer.front()->angular_velocity.y,
                              imu_rtk_buffer.front()->angular_velocity.z;

                acc_avr << imu_rtk_buffer.front()->linear_acceleration.x,
                           imu_rtk_buffer.front()->linear_acceleration.y,
                           imu_rtk_buffer.front()->linear_acceleration.z;

                rot_rtk = rot_rtk * Sophus::SO3d::exp(angvel_avr * dt_rtk); 
                acc_avr = rot_rtk * acc_avr + state_point_last.grav;
                vel_rtk += acc_avr * dt_rtk;   

                z_rtk[0] += vel_rtk[0] * dt_rtk;
                z_rtk[1] += vel_rtk[1] * dt_rtk;
                z_rtk[2] += vel_rtk[2] * dt_rtk;

                imu_rtk_buffer.pop_front();
            }

            rtk_vaild = true;
        }
    }

    //判断heading的可用性
    rtk_heading_vaild = false;
    while (!gnss_heading_buffer.empty())
    {
        // 删除当前帧0.05s之前的rtk
        if (heading_time_buffer.front() < Measures.package_end_time - 0.10)
        {
            gnss_heading_buffer.pop_front();
            heading_time_buffer.pop_front();
        }
        // 超过当前帧0.05s之后，退出
        else if (heading_time_buffer.front() > Measures.package_end_time)
        {
            break;
        }
        else
        {
            kf_heading = gnss_heading_buffer.front();

            gnss_heading_buffer.pop_front();
            heading_time_buffer.pop_front();

            rtk_heading_vaild = true;
        }
    }
}

void publish_sensor_vaild()
{
    na_localization::sensor_vaild SensorVaild;
    SensorVaild.header.stamp = ros::Time().fromSec(Measures.package_end_time);
    SensorVaild.imu_vaild = imu_vaild;
    SensorVaild.lidar_vaild = Measures.lidar_vaild;
    SensorVaild.rtk_vaild = rtk_vaild;
    SensorVaild.leg_vaild = Measures.leg_vaild;
    pubSensorVaild.publish(SensorVaild);
}

void publish_localization_vaild()
{
    std_msgs::Bool msg_localization;

    if(need_relocal)
    {
        msg_localization.data = false;
    }
    else
    {
        msg_localization.data = true;
    }

    pubLocalizationVaild.publish(msg_localization);
}

// BoxPointType LocalMap_Points;           // ikd-tree地图立方体的2个角点
// bool Localmap_Initialized = false;      // 局部地图是否初始化
void lasermap_fov_segment()
{
    cub_needrm.clear();     // 清空需要移除的区域
    V3D pos_LiD = state_point.pos;  // W系下位置
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
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        // 与某个方向上的边界距离（1.5*300m）太小，标记需要移除need_move(FAST-LIO2论文Fig.3)
        // if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) need_move = true;
        if (dist_to_map_edge[i][0] <= 80 || dist_to_map_edge[i][1] <= 80) need_move = true;
    }
    if (!need_move) return;  //如果不需要，直接返回，不更改局部地图
    cout<<"remove points"<<endl;
    double time_remove1 = omp_get_wtime();
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    //需要移动的距离
    // float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD -1)));
    float mov_dist = 20;
    for (int i = 0; i < 3; i++){
        tmp_boxpoints = LocalMap_Points;
        // if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE){
        if (dist_to_map_edge[i][0] <= 80){
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        // } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE){
        } else if (dist_to_map_edge[i][1] <= 80){
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }    
    LocalMap_Points = New_LocalMap_Points;

    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);

    if(cub_needrm.size() > 0) ikdtree.Delete_Point_Boxes(cub_needrm); //删除指定范围内的点

    double time_remove2 = omp_get_wtime();
    cout<<"remove time:"<<time_remove2 - time_remove1<<endl;
}

void map_incremental()
{
    double time_addpoint1 = omp_get_wtime();
    PointVector PointNoNeedDownsample;
    PointNoNeedDownsample.reserve(MM.pointcloud_add->points.size());
    for(int i=0;i<MM.pointcloud_add->points.size();i++)
    {
        PointNoNeedDownsample.push_back(MM.pointcloud_add->points[i]);
    }
    ikdtree.Add_Points(PointNoNeedDownsample, false); 
    double time_addpoint2 = omp_get_wtime();
    cout<<"points size:"<<MM.pointcloud_add->points.size()<<endl;
    cout<<"ikdtree size:"<<ikdtree.validnum()<<endl;
    cout<<"add point time:"<<time_addpoint2 - time_addpoint1<<endl;
}

//重构ikdtree线程
void ReikdtreeThread()
{
    // cout<<"******************"<<endl;
    ros::Rate rate1(1);
    while (ros::ok())
    {
        if (flg_exit)
            break;
        //如果需要重定位,continue
        if(need_relocal)
            continue;

        if (flag_ikdtree_initial==true)
        {

            pos_now.x=state_point.pos[0];
            pos_now.y=state_point.pos[1];
            pos_now.z=state_point.pos[2];

            /***********新增代码*************/
            if(MM.get_map(state_point.pos[0],state_point.pos[1]))
            {   
                mtx_reikd.lock();
                if((flag_thread1 == 0 || flag_thread1 == 2) && last_ikdtree == 2)
                {
                    flag_thread2 = 1;
                    mtx_reikd.unlock();

                    double time_update1 = omp_get_wtime();
                    // ikdtree.reconstruct(MM.pointcloud_output->points);
                    kdtree1.setInputCloud(MM.pointcloud_output);
                    double time_update2 = omp_get_wtime();
                    // cout<<"update1 time:"<<time_update2 - time_update1<<endl;

                    // pcl::KdTreeFLANN<PointType> kdtree_test;
                    // double time_kdtree1 = omp_get_wtime();
                    // kdtree_test.setInputCloud(MM.pointcloud_output);
                    // double time_kdtree2 = omp_get_wtime();
                    // cout<<"kdtree1 time:"<<time_kdtree2 - time_kdtree1<<endl;

                    mtx_reikd.lock();
                    flag_thread2 = 0;
                    last_ikdtree = 1;
                    mtx_reikd.unlock();
                }
                // else if(flag_thread1 == 1)
                else if((flag_thread1 == 0 || flag_thread1 == 1) && last_ikdtree == 1)
                {
                    flag_thread2 = 2;
                    mtx_reikd.unlock();

                    double time_update1 = omp_get_wtime();
                    // ikdtree2.reconstruct(MM.pointcloud_output->points);
                    kdtree2.setInputCloud(MM.pointcloud_output);
                    double time_update2 = omp_get_wtime();
                    // cout<<"update2 time:"<<time_update2 - time_update1<<endl;

                    // pcl::KdTreeFLANN<PointType> kdtree_test;
                    // double time_kdtree1 = omp_get_wtime();
                    // kdtree_test.setInputCloud(MM.pointcloud_output);
                    // double time_kdtree2 = omp_get_wtime();
                    // cout<<"kdtree2 time:"<<time_kdtree2 - time_kdtree1<<endl;

                    mtx_reikd.lock();
                    flag_thread2 = 0;
                    last_ikdtree = 2;
                    mtx_reikd.unlock();
                }
                else
                {
                    ROS_ERROR("Thread2 Error!");
                    mtx_reikd.unlock();
                }

            }
            /************************/

            // if(MM.get_map(state_point.pos[0],state_point.pos[1]))
            // {   
            //     mtx_reikd.lock();
            //     double time_update1 = omp_get_wtime();
            //     ikdtree.reconstruct(MM.pointcloud_output->points);
            //     double time_update2 = omp_get_wtime();
            //     cout<<"update time:"<<time_update2 - time_update1<<endl;
            //     mtx_reikd.unlock();
            // }

            // if(MM.get_map_add(state_point.pos[0],state_point.pos[1]))
            // {
            //     mtx_reikd.lock();
            //     // bool map_update_tmp = map_update;
            //     // mtx_reikd.unlock();

            //     // if(map_update_tmp == false)
            //     // {
            //     //     if(MM.need_relocal == true)
            //     //         need_relocal = true; 
            //     //     lasermap_fov_segment();
            //     //     map_incremental();

            //     //     // mtx_reikd.lock();
            //     //     map_update = true;
            //     //     // mtx_reikd.unlock();
            //     // }
            //     // mtx_reikd.unlock();
            //     // mtx_reikd.lock();
                
            //     // double time_update1 = omp_get_wtime();
            //     // flag_reikdtree=true;
                
            //     if(MM.need_relocal == true)
            //         need_relocal = true; 

            //     lasermap_fov_segment();
            //     map_incremental();
            //     // double time_update2 = omp_get_wtime();
            //     // cout<<"map update time:"<<time_update2 - time_update1<<endl;

            //     mtx_reikd.unlock();
            //     cout<<"Rebuild IKdtree Successful! current pos is: x:"<<pos_now.x<<" y:"<<pos_now.y<<" z:"<<pos_now.z<<endl;
            // }

            // if (pointDistance(pos_now, pos_last) > 30.0)
            // {
                    
            //     // pcl::PointCloud<PointType>::Ptr mapikdtree(new pcl::PointCloud<PointType>());
            //     mapikdtree->clear();

            //     for(int i=0;i<(int)pointcloudmap->size();i++)
            //     {
            //         if (pointDistance(pointcloudmap->points[i], pos_now) > 80.0)
            //         continue;
                        
            //         mapikdtree->push_back(pointcloudmap->points[i]);
            //     }
            //     // ikdtree.reconstruct(mapikdtree->points);
            //     pos_last=pos_now;

            //     mtx_reikd.lock();
            //     flag_reikdtree=true;
            //     mtx_reikd.unlock();
            //     cout<<"Rebuild IKdtree Successful! current pos is: x:"<<pos_now.x<<" y:"<<pos_now.y<<" z:"<<pos_now.z<<endl;
            //     // cout<<"Rebuild IKdtree Successful!"<<endl;
            // }
        }
        rate1.sleep();
    }  
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);                // 是否发布当前正在扫描的点云的topic
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);              // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic 
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);     // 是否发布经过运动畸变校正注册到IMU坐标系的点云的topic，需要该变量和上一个变量同时为true才发布
    nh.param<bool>("publish/visual_ikdtree", visual_ikdtree, true);              // 是否发布ikdtree点云
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
    nh.param<vector<double>>("mapping/rtk2Lidar_T", rtk2Lidar_T, vector<double>()); // rtk相对于雷达的外参T（即rtk在Lidar坐标系中的坐标）
    
    //rtk
    nh.param<bool>("usertk", usertk, false); //是否使用rtk
    nh.param<string>("common/gnss_topic", gnss_topic,"/rtk_pos_raw");   //gps的topic名称

    nh.param<string>("common/gnss_heading_topic", gnss_heading_topic,"/rtk_heading_raw");   //gps_heading的topic名称 //!

    nh.param<std::string>("loadmappath", loadmappath, "/home/ywb/NR_mapping/src/FAST_LIO_SAM/FAST_LIO_SAM/PCD/cloud_map.pcd"); //加载地图的路径
    nh.param<std::string>("loadposepath", loadposepath, "/home/ywb/NR_mapping/src/FAST_LIO_SAM/FAST_LIO_SAM/PCD/pose.txt"); //加载初始位置的路径（只有用rtk才需要）
  
    nh.param<string>("common/leg_topic", leg_topic,"/leg_odom");   //leg的topic名称
    nh.param<bool>("useleg",useleg,false); //是否使用leg

    nh.param<int>("Reposition_type", Reposition_type, 2);



    cout<<"Lidar_type: "<<p_pre->lidar_type<<endl;
    // 初始化path的header（包括时间戳和帧id），path用于保存odemetry的路径
    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 1, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 1, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Subscriber sub_gnss = nh.subscribe(gnss_topic, 200000, gnss_cbk); //gnss

    ros::Subscriber sub_gnss_heading = nh.subscribe(gnss_heading_topic, 200000, gnss_heading_cbk);   //!

    ros::Subscriber sub_leg = nh.subscribe(leg_topic, 200000, leg_cbk); 
    // ros::Subscriber sub_ExterPos=nh.subscribe("/move_base_simple/goal",1,ExtPos_cbk); //手动重定位话题
    ros::Subscriber sub_ManualPos=nh.subscribe("/move_base_simple/goal",1,ManualPos_cbk); //手动重定位话题
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry> ("/Odometry", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path> ("/path", 100000);
    ros::Publisher pubPathUpdate = nh.advertise<nav_msgs::Path>("s_fast_lio/path_update", 100000);                   //  isam更新后的path
    ros::Publisher pubGnssPath = nh.advertise<nav_msgs::Path>("/gnss_path", 100000);
    ros::Publisher pub_save = nh.advertise<std_msgs::Bool>("/save_data", 100); //发布保存数据的话题
    pubSensorVaild = nh.advertise<na_localization::sensor_vaild>("/sensor_vaild", 100000); //发布目前传感器的可用性
    pubLocalizationVaild = nh.advertise<std_msgs::Bool>("/localization_vaild", 1000); //发布定位的有效性
    //发布全局地图,用来手动重定位
    pubglobalmap =nh.advertise<sensor_msgs::PointCloud2>("/globalmap",1);
    //发布定位结果的个数，用来统计定位频率
    ros::Publisher pubOdoCnt = nh.advertise<std_msgs::Int32>("/odometry_count", 100);



    //todo 2024.1.15 修改日志内容 尝试删除日志相关代码
    //TODO 2024.1.16 所有关于 NLOG_INFO的内容都去除 整个代码文件里面 已经尽量去除了


    string params_filename = string("/home/ljq/alg_ywb/na_localization/src/na_localization/PCD/param.json");
    reloc_plugin_ptr_ = std::make_shared<plugins::RelocPlugin>(nh, params_filename);    
    /****************************/

    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);

    shared_ptr<ImuProcess> p_imu1(new ImuProcess());
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    rtk_T_wrt_Lidar<<VEC_FROM_ARRAY(rtk2Lidar_T);
    p_imu1->set_param(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU, V3D(gyr_cov, gyr_cov, gyr_cov), V3D(acc_cov, acc_cov, acc_cov), 
                        V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov), V3D(b_acc_cov, b_acc_cov, b_acc_cov));



    /*************************重定位****************************/
    string read_dir = loadmappath;
    pcl::PCDReader pcd_reader;
    pcd_reader.read(read_dir, *pointcloudmap);
    cout<<"read pcd success!"<<endl;

    pcl::VoxelGrid<PointType> downSizepointcloudmap;
    pcl::PointCloud<PointType>::Ptr DSpointcloudmap(new pcl::PointCloud<PointType>());//地图点云

    downSizepointcloudmap.setInputCloud(pointcloudmap);
    downSizepointcloudmap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    downSizepointcloudmap.filter(*pointcloudmap);

    //需要对地图点云降采样，不然在rviz里显示太卡
    downSizepointcloudmap.setInputCloud(pointcloudmap);
    downSizepointcloudmap.setLeafSize(1.0f, 1.0f, 1.0f);
    downSizepointcloudmap.filter(*DSpointcloudmap);

    // sensor_msgs::PointCloud2 globalmapmsg;
    pcl::toROSMsg(*DSpointcloudmap, globalmapmsg);
    globalmapmsg.header.frame_id = "camera_init"; //todo 这里发布一个从读取点云里面稀疏后的降采样点云 就是为了方便在RVIZ里面观察

    // if(usertk)
    // {
    //     //读取原点对应的经纬高坐标
    //     FILE *fp1;
    //     fp1=fopen(loadposepath.c_str(),"r");
    //     fscanf(fp1,"%lf %lf %lf",&lat0,&lon0,&alt0);
    //     fclose(fp1); 
    //     // cout<<"lat0:"<<lat0<<" lon0:"<<lon0<<" alt0:"<<alt0<<endl;
    //     printf("%2.10lf %3.10f %2.5lf",lat0,lon0,alt0);
    //     fflush(stdout);
    //     //gps位置初始化
    //     gnss_data.InitOriginPosition(lat0, lon0, alt0) ; 
    // }
    /**********************************************************/

    /****************加载地图******************/
    // map_management MM;
    MM.set_ds_size(filter_size_map_min);
    MM.set_input_PCD(loadmappath);
    MM.voxel_process();
    /****************************************/
    //重构ikd树的线程
    std::thread ikdtreethread(&ReikdtreeThread);
    
    ros::Rate ratemap(1); //todo 接上文 发布了一个在RVIZ里面方便观察的稀疏地图 （源自读取）要延迟1S 发布在RVIZ中 否则会RVIZ收不到
    ratemap.sleep();
    pubglobalmap.publish(globalmapmsg);
    // ratemap.sleep();
    
    Eigen::Matrix3d Sigma_leg = Eigen::Matrix3d::Identity(); //leg里程计的协方差
    double sigmaleg = 0.0025;//0.01
    Sigma_leg(0, 0) = sigmaleg;
    Sigma_leg(1, 1) = sigmaleg;
    Sigma_leg(2, 2) = sigmaleg;

    Eigen::Matrix3d Sigma_rtk = Eigen::Matrix3d::Identity(); //rtk的协方差
    double sigmartk = 0.05*0.05;
    Sigma_rtk(0, 0) = sigmartk;
    Sigma_rtk(1, 1) = sigmartk;
    Sigma_rtk(2, 2) = sigmartk;

    Eigen::Vector3d z_leg = Eigen::Vector3d::Zero();
    Eigen::Vector3d z_rtk = Eigen::Vector3d::Zero();

    bool save_data = true;

    double odo_time = 0;
    int odo_cnt = 0;

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);

    while (ros::ok())
    {
        // std::cout << "0.1" << std::endl;
        if (flg_exit) break;
        ros::spinOnce();

        //重定位
        ReLocalization();


        if(sync_packages(Measures))  //把一次的IMU和LIDAR数据打包到Measures
        {

            // if(p_pre->get_avg_range() < 1.5)
            // {
            //     filter_size_surf_min = 0.1;
            // }
            // else if(p_pre->get_avg_range() < 4.0)
            // {
            //     filter_size_surf_min = 0.2;
            // }
            // else
            // {
            //     filter_size_surf_min = 0.4;
            // }

            // cout<<"filter_size_surf_min:"<<filter_size_surf_min<<endl;

            //如果需要重定位，continue
            if(need_relocal==true)
            {
                continue;
            }

            double time_all1 = omp_get_wtime();
            double t00 = omp_get_wtime();

            if (flg_first_scan)
            {
                
                first_lidar_time = Measures.lidar_beg_time;
                p_imu1->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                
                continue;
            }
            
            state_point_last = state_point;
            double t_process1 = omp_get_wtime();
            p_imu1->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            double t_process2 = omp_get_wtime();
            // cout<<"process time:"<<t_process2 - t_process1<<endl;

            if(!p_imu1->feats_undistort_vaild)
                continue;
            
            // //如果feats_undistort为空 ROS_WARN
            // if (feats_undistort->points.size() < 50 || (feats_undistort == NULL))
            // {
            //     ROS_WARN("No point, skip this scan!\n");
            //     continue;
            // }

            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? false : true;

            // lasermap_fov_segment();
            //点云下采样
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            feats_down_size = feats_down_body->points.size();
            
            double time_all2 = omp_get_wtime();
            // if(ikdtree.Root_Node == nullptr)
            // {
            //     ikdtree.set_downsample_param(filter_size_map_min);

            //     PointType pos_initial;
            //     pos_initial.x=state_point.pos[0];
            //     pos_initial.y=state_point.pos[1];
            //     pos_initial.z=state_point.pos[2];

            //     pos_last=pos_initial;

            //     mapikdtree->clear();

            //     for(int i=0;i<(int)pointcloudmap->size();i++)
            //     {
            //         if (pointDistance(pointcloudmap->points[i], pos_initial) > 80.0)
            //         continue;
                
            //         mapikdtree->push_back(pointcloudmap->points[i]);
            //     }

            //     flag_ikdtree_initial=true;

            //     ikdtree.Build(mapikdtree->points);
            // }

            // if(ikdtree.Root_Node == nullptr)
            if(flag_ikdtree_initial == false)
            {
                // ikdtree.set_downsample_param(filter_size_map_min);
                // ikdtree2.set_downsample_param(filter_size_map_min);

                PointType pos_initial;
                pos_initial.x=state_point.pos[0];
                pos_initial.y=state_point.pos[1];
                pos_initial.z=state_point.pos[2];
                //获得初始的局部地图
                MM.get_map(state_point.pos[0],state_point.pos[1]);

                pos_last=pos_initial;

                flag_ikdtree_initial=true;

                kdtree1.setInputCloud(MM.pointcloud_output);
                kdtree2.setInputCloud(MM.pointcloud_output);
                // ikdtree.Build(MM.pointcloud_output->points);
                // ikdtree2.Build(MM.pointcloud_output->points);

            }
            
            //判断传感器的可用性
            sensor_vaild_judge(z_leg,z_rtk);

            //计算当前roll、pitch、yaw
            Eigen::Vector3d cur_atti = Eigen::Vector3d::Zero();
            cur_atti[0] = atan2(state_point.rot.matrix()(2,1),state_point.rot.matrix()(2,2));
            cur_atti[1] = -asin(state_point.rot.matrix()(2,0));
            cur_atti[2] = atan2(state_point.rot.matrix()(1,0),state_point.rot.matrix()(0,0));

            // cout<<"roll:"<<cur_atti[0]*57.3<<" pitch:"<<cur_atti[1]*57.3<<" yaw:"<<cur_atti[2]*57.3<<endl;
            /*** iterated state estimation ***/
            Nearest_Points.resize(feats_down_size);         //存储近邻点的vector
            // kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_body, ikdtree, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en);
            // if(Measures.lidar_vaild == true || Measures.leg_vaild == true)
            // {
            //     kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_body, ikdtree, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en,
            //                                       Sigma_leg,z_leg,useleg,Measures.leg_vaild,Measures.lidar_vaild,need_relocal,Sigma_rtk,z_rtk,rtk_vaild,cur_atti,kf_heading,rtk_heading_vaild);
            // }
            rtk_vaild = false;
            rtk_heading_vaild = false;
            // Measures.lidar_vaild = false;

            // double time_all2 = omp_get_wtime();
            /************新代码***************/
            mtx_reikd.lock();
            if((flag_thread2 == 0 && last_ikdtree == 1) || flag_thread2 == 2)
            {
                flag_thread1 = 1;
                mtx_reikd.unlock();
                

                if(visual_ikdtree)
                {
                    featsFromMap->clear();
                    // double time_kd1 = omp_get_wtime();
                    *featsFromMap = *kdtree1.getInputCloud();
                    // double time_kd2 = omp_get_wtime();
                    // cout<<"kd1 time:"<<time_kd2 - time_kd1<<endl;
                }
                double time_kf1 = omp_get_wtime();
                kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_body, kdtree1, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en,
                                                  Sigma_leg,z_leg,useleg,Measures.leg_vaild,Measures.lidar_vaild,need_relocal,Sigma_rtk,z_rtk,rtk_vaild,cur_atti,kf_heading,rtk_heading_vaild);
                double time_kf2 = omp_get_wtime();
                // if((time_kf2 - time_kf1) > 0.8)
                // {
                // 	cout<<"kf1 time:"<<time_kf2 - time_kf1<<endl;
                // }
                

                mtx_reikd.lock();
                flag_thread1 = 0;
                mtx_reikd.unlock();
            }
            else if((flag_thread2 == 0 && last_ikdtree == 2) || flag_thread2 == 1)
            {
                flag_thread1 = 2;
                mtx_reikd.unlock();

                if(visual_ikdtree)
                {
                    featsFromMap->clear();
                    // double time_kd1 = omp_get_wtime();
                    *featsFromMap = *kdtree2.getInputCloud();
                    // double time_kd2 = omp_get_wtime();
                    // cout<<"kd2 time:"<<time_kd2 - time_kd1<<endl;
                }
                double time_kf1 = omp_get_wtime();
                kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_body, kdtree2, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en,
                                                  Sigma_leg,z_leg,useleg,Measures.leg_vaild,Measures.lidar_vaild,need_relocal,Sigma_rtk,z_rtk,rtk_vaild,cur_atti,kf_heading,rtk_heading_vaild);
                double time_kf2 = omp_get_wtime();
                // if((time_kf2 - time_kf1) > 0.8)
                // {
                // 	cout<<"kf2 time:"<<time_kf2 - time_kf1<<endl;
                // }
                

                mtx_reikd.lock();
                flag_thread1 = 0;
                mtx_reikd.unlock();
            }
            else
            {
                ROS_ERROR("Thread2 Error!");
                mtx_reikd.unlock();
            }
            /***************************/
            double time_all3 = omp_get_wtime();
            // mtx_reikd.lock();
            // if(visual_ikdtree)
            // {
            //     PointVector ().swap(ikdtree.PCL_Storage);
            //     ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
            //     featsFromMap->clear();
            //     featsFromMap->points = ikdtree.PCL_Storage;
            // }
            // // if(visual_ikdtree)
            // // {
            // //     PointVector().swap(ikdtree_tmp.PCL_Storage);
            // //     ikdtree_tmp.flatten(ikdtree_tmp.Root_Node, ikdtree_tmp.PCL_Storage, NOT_RECORD);
            // //     featsFromMap->clear();
            // //     featsFromMap->points = ikdtree_tmp.PCL_Storage;
            // // }
            // // mtx_reikd.lock();
            // // // if(map_update)
            // // {
            // //     ikdtree_tmp = ikdtree;
            // //     map_update = false;
            // // }
            // // mtx_reikd.unlock();

            // kf.update_iterated_dyn_share_modified(LASER_POINT_COV, feats_down_body, ikdtree, Nearest_Points, NUM_MAX_ITERATIONS, extrinsic_est_en,
            //                                       Sigma_leg,z_leg,useleg,Measures.leg_vaild,Measures.lidar_vaild,need_relocal,Sigma_rtk,z_rtk,rtk_vaild,cur_atti,kf_heading,rtk_heading_vaild);
            // mtx_reikd.unlock();
            

            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot.matrix() * state_point.offset_T_L_I;

            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);
            //测试定位频率
            if(0 == odo_time)
            {
                odo_time = ros::Time::now().toSec();
                odo_cnt ++;
            }
            else
            {
                odo_cnt ++;
                if((ros::Time::now().toSec() - odo_time) > 10.0)
                {
                    std_msgs::Int32 odo_cnt_;
                    odo_cnt_.data = odo_cnt;
                    pubOdoCnt.publish(odo_cnt_);

                    odo_time = ros::Time::now().toSec();
                    odo_cnt = 0;
                }
            }

            /*** add the feature points to map kdtree ***/
            feats_down_world->resize(feats_down_size);
            
            /******* Publish points *******/
            if (path_en)
            {
                publish_path(pubPath);
                publish_path_update(pubPathUpdate);             //   发布经过isam2优化后的路径
                publish_gnss_path(pubGnssPath);                        //   发布gnss轨迹
            }                         
            if (scan_pub_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            if (visual_ikdtree)   publish_map(pubLaserCloudMap);
            
            // cout<<"555"<<endl;
            double t11 = omp_get_wtime();
            // std::cout << "feats_down_size: " <<  feats_down_size << "  Whole mapping time(ms):  " << (t11 - t00)*1000 << std::endl<< std::endl;
            double time_all4 = omp_get_wtime();
            //cout<<"all time:"<<time_all4 - time_all1<<endl;
            //cout<<"feats_down_size"<<feats_down_size<<endl;
            //cout<<"t1:"<<time_all2 - time_all1<<" t2:"<<time_all3 - time_all2<<"t3:"<<time_all4 - time_all3<<endl;
            double yaw = atan2(state_point.rot.matrix()(1,0),state_point.rot.matrix()(0,0)) * 180 / M_PI;
            // cout<<"lidar heading:"<<yaw<<" rtk heading:"<<rtk_heading* 180 / M_PI<<endl;
            // cout<<"delta = "<<yaw + rtk_heading* 180 / M_PI - 360.0<<endl;
            
        }

        rate.sleep();


        //发布传感器的有效性
        publish_sensor_vaild();
        //发布定位的有效性
        publish_localization_vaild();  

        if(need_relocal)
        {
            if(save_data == true)
            {
                std_msgs::Bool tmp_save_data ;
                tmp_save_data.data = true;
                pub_save.publish(tmp_save_data);
                save_data = false;
            }
        }
        else
        {
            save_data = true;
        }

        // mtx_reikd.lock();       
        // if (flag_reikdtree==true)
        // {
        //     ikdtree.reconstruct(MM.pointcloud_output->points);
        //     flag_reikdtree=false;
        // }
        // mtx_reikd.unlock();

        // mtx_reikd.lock();     
        // if(MM.need_relocal == true)
        //     need_relocal = true;  

        // if (flag_reikdtree==true)
        // {
        //     map_incremental();
        //     flag_reikdtree=false;
        // }
        // mtx_reikd.unlock();

        // rate.sleep();
    }

    ikdtreethread.join();

    return 0;
}
