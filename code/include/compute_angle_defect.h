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

        // Compute edge vectors
        Vector3d e0 = p1 - p0; // edge from v0 to v1
        Vector3d e1 = p2 - p1; // edge from v1 to v2
        Vector3d e2 = p0 - p2; // edge from v2 to v0

        // Compute squared edge lengths
        double l0_sqr = e0.squaredNorm();
        double l1_sqr = e1.squaredNorm();
        double l2_sqr = e2.squaredNorm();

        // Compute angles using law of cosines
        // angle = arccos((a² + b² - c²) / (2ab))
        // where a and b are adjacent edges and c is the opposite edge

        // Angle at vertex v0
        double cos_angle0 = (l0_sqr + (-e2).squaredNorm() - l1_sqr) /
                           (2.0 * sqrt(l0_sqr) * sqrt((-e2).squaredNorm()));
        cos_angle0 = std::max(-1.0, std::min(1.0, cos_angle0)); // Clamp to avoid numerical issues
        double angle0 = acos(cos_angle0);

        // Angle at vertex v1
        double cos_angle1 = (l0_sqr + l1_sqr - (-e2).squaredNorm()) /
                           (2.0 * sqrt(l0_sqr) * sqrt(l1_sqr));
        cos_angle1 = std::max(-1.0, std::min(1.0, cos_angle1)); // Clamp to avoid numerical issues
        double angle1 = acos(cos_angle1);

        // Angle at vertex v2
        double cos_angle2 = (l1_sqr + (-e2).squaredNorm() - l0_sqr) /
                           (2.0 * sqrt(l1_sqr) * sqrt((-e2).squaredNorm()));
        cos_angle2 = std::max(-1.0, std::min(1.0, cos_angle2)); // Clamp to avoid numerical issues
        double angle2 = acos(cos_angle2);

        // Accumulate angles at each vertex
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

