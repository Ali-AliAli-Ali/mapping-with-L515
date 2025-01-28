#include <iostream>
#include <librealsense2/rs.hpp>
#include <librealsense2-net/rs_net.hpp>

using namespace std;


int main(int argc, char *argv[])
{
    string geobot_ip = "172.20.10.10";
    cout << geobot_ip;
    try {
        rs2::net_device dev(geobot_ip);
        rs2::context ctx;
        dev.add_to(ctx);
    }
    catch (string err) {
        cout << err;
    }

//    cout << "librealsense version - " << RS2_API_VERSION_STR << std::endl;
//    cout << "You have " << ctx.query_devices().size() << " RealSense devices connected" << std::endl;


    return 0;
}
