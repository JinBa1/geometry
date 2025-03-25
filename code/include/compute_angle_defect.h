#ifndef COMPUTE_ANGLE_DEFECT_HEADER_FILE
#define COMPUTE_ANGLE_DEFECT_HEADER_FILE

#include <Eigen/Dense>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

Eigen::VectorXd compute_angle_defect(const Eigen::MatrixXd& V,
                                     const Eigen::MatrixXi& F,
                                     const Eigen::VectorXi& boundVMask){
    
    using namespace Eigen;
    using namespace std;

    // Initialize the result vector G with zeros
    VectorXd G = VectorXd::Zero(V.rows());

    // For each face
    for (int f = 0; f < F.rows(); ++f) {
        // Get the vertex indices for this face
        int v0 = F(f, 0);
        int v1 = F(f, 1);
        int v2 = F(f, 2);

        // Get vertex positions
        Vector3d p0 = V.row(v0);
        Vector3d p1 = V.row(v1);
        Vector3d p2 = V.row(v2);

        // Compute edges
        Vector3d e0 = p1 - p0;
        Vector3d e1 = p2 - p1;
        Vector3d e2 = p0 - p2;

        // Compute angles at each vertex using dot products and norms
        double angle0 = acos((-e0).dot(-e2) / (e0.norm() * e2.norm()));
        double angle1 = acos((-e1).dot(-e0) / (e1.norm() * e0.norm()));
        double angle2 = acos((-e2).dot(-e1) / (e2.norm() * e1.norm()));

        // Add the angles to their respective vertices
        G(v0) += angle0;
        G(v1) += angle1;
        G(v2) += angle2;
    }

    // Compute the angle defect for each vertex
    for (int v = 0; v < V.rows(); ++v) {
        if (boundVMask(v) == 1) {
            // Boundary vertex: G = pi - sum of angles
            G(v) = M_PI - G(v);
        } else {
            // Interior vertex: G = 2pi - sum of angles
            G(v) = 2.0 * M_PI - G(v);
        }
    }
    return G;
}


#endif
