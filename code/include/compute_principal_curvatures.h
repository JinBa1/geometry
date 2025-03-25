#ifndef COMPUTE_PRINCIPAL_CURVATURES_HEADER_FILE
#define COMPUTE_PRINCIPAL_CURVATURES_HEADER_FILE

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <cmath>

/**
 * Computes principal curvatures and directions at each vertex of a mesh by
 * fitting a quadratic function to the 1-ring neighborhood.
 *
 * @param V Vertex positions
 * @param F Face indices
 * @param k1 Output principal curvature 1 (maximum)
 * @param k2 Output principal curvature 2 (minimum)
 * @param d1 Output principal direction 1 (corresponds to k1)
 * @param d2 Output principal direction 2 (corresponds to k2)
 */
void compute_principal_curvatures(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::VectorXd& k1,
    Eigen::VectorXd& k2,
    Eigen::MatrixXd& d1,
    Eigen::MatrixXd& d2)
{
    using namespace Eigen;
    using namespace std;

    const int numVertices = V.rows();

    // Initialize output arrays
    k1 = VectorXd::Zero(numVertices);
    k2 = VectorXd::Zero(numVertices);
    d1 = MatrixXd::Zero(numVertices, 3);
    d2 = MatrixXd::Zero(numVertices, 3);

    // Compute vertex normals by averaging face normals
    vector<vector<int>> vertexFaces(numVertices);
    for (int f = 0; f < F.rows(); ++f) {
        for (int j = 0; j < 3; ++j) {
            vertexFaces[F(f, j)].push_back(f);
        }
    }

    MatrixXd vertexNormals = MatrixXd::Zero(numVertices, 3);

    // Compute face normals and accumulate for vertex normals
    for (int f = 0; f < F.rows(); ++f) {
        int v0 = F(f, 0);
        int v1 = F(f, 1);
        int v2 = F(f, 2);

        Vector3d p0 = V.row(v0);
        Vector3d p1 = V.row(v1);
        Vector3d p2 = V.row(v2);

        Vector3d normal = (p1 - p0).cross(p2 - p0);
        double area = normal.norm() * 0.5;
        normal.normalize();

        // Weight by face area
        vertexNormals.row(v0) += area * normal;
        vertexNormals.row(v1) += area * normal;
        vertexNormals.row(v2) += area * normal;
    }

    // Normalize vertex normals
    for (int v = 0; v < numVertices; ++v) {
        vertexNormals.row(v).normalize();
    }

    // Find 1-ring neighbors for each vertex
    vector<vector<int>> vertexNeighbors(numVertices);
    for (int f = 0; f < F.rows(); ++f) {
        for (int j = 0; j < 3; ++j) {
            int v0 = F(f, j);
            int v1 = F(f, (j+1)%3);

            // Add vertices to each other's neighbor lists if not already present
            if (find(vertexNeighbors[v0].begin(), vertexNeighbors[v0].end(), v1) == vertexNeighbors[v0].end()) {
                vertexNeighbors[v0].push_back(v1);
            }
            if (find(vertexNeighbors[v1].begin(), vertexNeighbors[v1].end(), v0) == vertexNeighbors[v1].end()) {
                vertexNeighbors[v1].push_back(v0);
            }
        }
    }

    // Process each vertex
    for (int v = 0; v < numVertices; ++v) {
        // Skip vertices with no neighbors
        if (vertexNeighbors[v].empty()) {
            continue;
        }

        // Get vertex position and normal
        Vector3d origin = V.row(v);
        Vector3d normal = vertexNormals.row(v);

        // Create local coordinate system
        Vector3d zAxis = normal;

        // Find a perpendicular vector for xAxis
        Vector3d xAxis;
        if (abs(zAxis(0)) < abs(zAxis(1)) && abs(zAxis(0)) < abs(zAxis(2))) {
            xAxis = Vector3d(1, 0, 0).cross(zAxis);
        } else {
            xAxis = Vector3d(0, 1, 0).cross(zAxis);
        }
        xAxis.normalize();

        // yAxis completes the orthogonal basis
        Vector3d yAxis = zAxis.cross(xAxis);
        yAxis.normalize();

        // Create rotation matrix from world to local coordinates
        Matrix3d rotation;
        rotation.row(0) = xAxis;
        rotation.row(1) = yAxis;
        rotation.row(2) = zAxis;

        // Get 1-ring in local coordinates
        MatrixXd localPoints(vertexNeighbors[v].size() + 1, 3);

        // Add central vertex (will be at origin in local coordinates)
        Vector3d localOrigin = rotation * (origin - origin);
        localPoints.row(0) = localOrigin;

        // Add neighbors
        for (size_t i = 0; i < vertexNeighbors[v].size(); ++i) {
            int neighbor = vertexNeighbors[v][i];
            Vector3d neighborPos = V.row(neighbor);
            Vector3d localPos = rotation * (neighborPos - origin);
            localPoints.row(i + 1) = localPos;
        }

        // Prepare the least squares system
        int n = localPoints.rows();
        MatrixXd A(n, 6);
        VectorXd b(n);

        // Build the system: z = ax² + by² + cxy + dx + ey + f
        for (int i = 0; i < n; ++i) {
            double x = localPoints(i, 0);
            double y = localPoints(i, 1);
            double z = localPoints(i, 2);

            A(i, 0) = x * x;    // a
            A(i, 1) = y * y;    // b
            A(i, 2) = x * y;    // c
            A(i, 3) = x;        // d
            A(i, 4) = y;        // e
            A(i, 5) = 1.0;      // f

            b(i) = z;
        }

        // Solve the least squares problem
        VectorXd coeffs;
        if (n >= 6) {
            // Standard least squares
            coeffs = A.colPivHouseholderQr().solve(b);
        } else {
            // Minimum norm solution if not enough points
            coeffs = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);
        }

        // Extract quadratic coefficients
        double a = coeffs(0);
        double b_coeff = coeffs(1);
        double c = coeffs(2);

        // Form the Hessian matrix
        Matrix2d hessian;
        hessian << 2*a, c,
                   c, 2*b_coeff;

        // Compute eigenvalues and eigenvectors of the Hessian
        SelfAdjointEigenSolver<Matrix2d> solver(hessian);
        Vector2d eigenvalues = solver.eigenvalues();
        Matrix2d eigenvectors = solver.eigenvectors();

        // Sort eigenvalues (k1 > k2) and corresponding eigenvectors
        if (eigenvalues(0) > eigenvalues(1)) {
            k1(v) = eigenvalues(0);
            k2(v) = eigenvalues(1);

            // Convert eigenvectors from local 2D coordinates to global 3D coordinates
            Vector3d dir1_local(eigenvectors(0, 0), eigenvectors(1, 0), 0);
            Vector3d dir2_local(eigenvectors(0, 1), eigenvectors(1, 1), 0);

            // Convert back to world coordinates
            Vector3d dir1_world = rotation.transpose() * dir1_local;
            Vector3d dir2_world = rotation.transpose() * dir2_local;

            d1.row(v) = dir1_world.normalized();
            d2.row(v) = dir2_world.normalized();
        } else {
            k1(v) = eigenvalues(1);
            k2(v) = eigenvalues(0);

            // Convert eigenvectors from local 2D coordinates to global 3D coordinates
            Vector3d dir1_local(eigenvectors(0, 1), eigenvectors(1, 1), 0);
            Vector3d dir2_local(eigenvectors(0, 0), eigenvectors(1, 0), 0);

            // Convert back to world coordinates
            Vector3d dir1_world = rotation.transpose() * dir1_local;
            Vector3d dir2_world = rotation.transpose() * dir2_local;

            d1.row(v) = dir1_world.normalized();
            d2.row(v) = dir2_world.normalized();
        }

        // Ensure principal directions are perpendicular to normal
        // Project out any component along the normal
        Vector3d normal_vec = vertexNormals.row(v);
        d1.row(v) = (d1.row(v) - d1.row(v).dot(normal_vec) * normal_vec).normalized();
        d2.row(v) = (d2.row(v) - d2.row(v).dot(normal_vec) * normal_vec).normalized();
    }

    return;
}

#endif