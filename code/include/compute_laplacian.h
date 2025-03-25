#ifndef COMPUTE_LAPLACIAN_HEADER_FILE
#define COMPUTE_LAPLACIAN_HEADER_FILE

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <cmath>

// Helper function to compute cotangent of the angle between two vectors
double cotangent(const Eigen::Vector3d& a, const Eigen::Vector3d& b) {
    double dot = a.dot(b);
    Eigen::Vector3d cross = a.cross(b);
    double sin_theta = cross.norm();

    // Check for numerical stability
    if (sin_theta < 1e-10) {
        return 0.0; // Default to zero for degenerate cases
    }

    return dot / sin_theta;
}

void compute_laplacian(const Eigen::MatrixXd& V,
                       const Eigen::MatrixXi& F,
                       const Eigen::MatrixXi& E,
                       const Eigen::MatrixXi& EF,
                       const Eigen::VectorXi& boundEMask,
                       Eigen::SparseMatrix<double>& d0,
                       Eigen::SparseMatrix<double>& W,
                       Eigen::VectorXd& vorAreas){

    using namespace Eigen;
    using namespace std;

    // Initialize matrices
    d0.resize(E.rows(), V.rows());
    W.resize(E.rows(), E.rows());
    vorAreas = VectorXd::Zero(V.rows());

    // Compute d0 (differential operator)
    vector<Triplet<double>> d0Triplets;
    for (int e = 0; e < E.rows(); ++e) {
        int source = E(e, 0);
        int target = E(e, 1);
        d0Triplets.push_back(Triplet<double>(e, source, -1.0));
        d0Triplets.push_back(Triplet<double>(e, target, 1.0));
    }
    d0.setFromTriplets(d0Triplets.begin(), d0Triplets.end());

    // Calculate face areas and contribute to Voronoi areas
    for (int f = 0; f < F.rows(); ++f) {
        int v0 = F(f, 0);
        int v1 = F(f, 1);
        int v2 = F(f, 2);

        Vector3d p0 = V.row(v0);
        Vector3d p1 = V.row(v1);
        Vector3d p2 = V.row(v2);

        Vector3d e01 = p1 - p0;
        Vector3d e02 = p2 - p0;

        double area = 0.5 * (e01.cross(e02)).norm();

        // Add area contribution to each vertex (using barycentric coordinates)
        vorAreas(v0) += area / 3.0;
        vorAreas(v1) += area / 3.0;
        vorAreas(v2) += area / 3.0;
    }

    // Compute W (cotangent weights)
    vector<Triplet<double>> WTriplets;

    for (int e = 0; e < E.rows(); ++e) {
        double weight = 0.0;

        // Process the left face
        if (EF(e, 0) != -1) {
            int faceIdx = EF(e, 0);
            int oppVertIdx = EF(e, 1);

            // Get vertices of the face
            int v0 = F(faceIdx, 0);
            int v1 = F(faceIdx, 1);
            int v2 = F(faceIdx, 2);

            // Find opposite vertex
            int oppVertex;
            if (oppVertIdx == 0) oppVertex = v0;
            else if (oppVertIdx == 1) oppVertex = v1;
            else oppVertex = v2;

            int source = E(e, 0);
            int target = E(e, 1);

            // Find vectors for cotangent calculation
            Vector3d pOpp = V.row(oppVertex);
            Vector3d pSource = V.row(source);
            Vector3d pTarget = V.row(target);

            Vector3d vecA = pOpp - pSource;
            Vector3d vecB = pOpp - pTarget;

            // Compute cotangent
            double cotAlpha = cotangent(vecA, vecB);
            weight += 0.5 * cotAlpha;
        }

        // Process the right face if it exists (not a boundary edge)
        if (boundEMask(e) == 0 && EF(e, 2) != -1) {
            int faceIdx = EF(e, 2);
            int oppVertIdx = EF(e, 3);

            // Get vertices of the face
            int v0 = F(faceIdx, 0);
            int v1 = F(faceIdx, 1);
            int v2 = F(faceIdx, 2);

            // Find opposite vertex
            int oppVertex;
            if (oppVertIdx == 0) oppVertex = v0;
            else if (oppVertIdx == 1) oppVertex = v1;
            else oppVertex = v2;

            int source = E(e, 0);
            int target = E(e, 1);

            // Find vectors for cotangent calculation
            Vector3d pOpp = V.row(oppVertex);
            Vector3d pSource = V.row(source);
            Vector3d pTarget = V.row(target);

            Vector3d vecA = pOpp - pTarget;
            Vector3d vecB = pOpp - pSource;

            // Compute cotangent
            double cotBeta = cotangent(vecA, vecB);
            weight += 0.5 * cotBeta;
        }

        // Add weight to the diagonal of W
        WTriplets.push_back(Triplet<double>(e, e, weight));
    }

    W.setFromTriplets(WTriplets.begin(), WTriplets.end());

    return;
}

#endif