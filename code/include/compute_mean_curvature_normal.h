#ifndef COMPUTE_MEAN_CURVATURE_NORMAL_HEADER_FILE
#define COMPUTE_MEAN_CURVATURE_NORMAL_HEADER_FILE

#include <Eigen/Dense>
#include <Eigen/Sparse>

void compute_mean_curvature_normal(const Eigen::MatrixXd& V,
                                   const Eigen::MatrixXi& F,
                                   const Eigen::SparseMatrix<double>& L,
                                   const Eigen::VectorXd& vorAreas,
                                   Eigen::MatrixXd& Hn,
                                   Eigen::VectorXd& H){
    using namespace Eigen;
    using namespace std;

    // Initialize output matrices
    const int numVertices = V.rows();
    Hn = MatrixXd::Zero(numVertices, 3);
    H = VectorXd::Zero(numVertices);

    // Calculate per-vertex normals first
    MatrixXd vertexNormals = MatrixXd::Zero(numVertices, 3);

    // For each face, compute its normal and add to each vertex
    for (int f = 0; f < F.rows(); ++f) {
        int v0 = F(f, 0);
        int v1 = F(f, 1);
        int v2 = F(f, 2);

        Vector3d p0 = V.row(v0);
        Vector3d p1 = V.row(v1);
        Vector3d p2 = V.row(v2);

        // Compute face normal
        Vector3d e1 = p1 - p0;
        Vector3d e2 = p2 - p0;
        Vector3d normal = e1.cross(e2);

        // Add to vertex normals
        vertexNormals.row(v0) += normal;
        vertexNormals.row(v1) += normal;
        vertexNormals.row(v2) += normal;
    }

    // Normalize vertex normals
    for (int v = 0; v < numVertices; ++v) {
        double norm = vertexNormals.row(v).norm();
        if (norm > 1e-10) {
            vertexNormals.row(v) /= norm;
        }
    }

    // Compute mean curvature normal: Hn = Lv/(2A)
    // First step: apply Laplacian to vertex positions
    MatrixXd LV = L * V;

    // Second step: Divide by twice the Voronoi area
    for (int v = 0; v < numVertices; ++v) {
        if (vorAreas(v) > 1e-10) {
            Hn.row(v) = LV.row(v) / (2.0 * vorAreas(v));
        }
    }

    // Compute signed mean curvature
    for (int v = 0; v < numVertices; ++v) {
        // Compute magnitude of mean curvature normal
        double magnitude = Hn.row(v).norm();

        // Determine sign by dot product with vertex normal
        double dotProduct = Hn.row(v).dot(vertexNormals.row(v));

        // Apply sign
        double sign = (dotProduct >= 0) ? 1.0 : -1.0;
        H(v) = sign * magnitude;
    }

    return;
}

#endif
