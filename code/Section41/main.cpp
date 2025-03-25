#include <polyscope/polyscope.h>
#include <polyscope/surface_mesh.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <vector>
#include <set>
#include <array>
#include <queue>
#include "readOFF.h"
#include "create_edge_list.h"
#include "compute_angle_defect.h"
#include "compute_laplacian.h"
#include "compute_mean_curvature_normal.h"
#include "compute_principal_curvatures.h" // Include our new header
#include <chrono>
#include <filesystem>

using namespace Eigen;
using namespace std;

MatrixXi F, E, EF;
VectorXi boundEMask, boundVMask, boundVertices;
MatrixXd V, Hn;
SparseMatrix<double> d0, W;
VectorXd vorAreas, H;
VectorXd k1, k2;     // Principal curvatures (k1 > k2)
MatrixXd d1, d2;     // Principal directions

int main()
{
    readOFF(DATA_PATH "/bunny.off", V, F);
    create_edge_list(F, E, EF, boundEMask, boundVMask, boundVertices);

    polyscope::init();
    polyscope::SurfaceMesh* psMesh = polyscope::registerSurfaceMesh("Mesh", V, F);

    // Compute angle defect (Gaussian curvature)
    auto start = std::chrono::high_resolution_clock::now();
    VectorXd G = compute_angle_defect(V, F, boundVMask);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Angle defect computation took " << (double)(duration.count())/1000.0 << " seconds to execute." << std::endl;

    // Compute Laplacian and Voronoi areas
    start = std::chrono::high_resolution_clock::now();
    compute_laplacian(V, F, E, EF, boundEMask, d0, W, vorAreas);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Laplacian computation took " << (double)(duration.count())/1000.0 << " seconds to execute." << std::endl;

    // Compute mean curvature and mean curvature normal
    start = std::chrono::high_resolution_clock::now();
    compute_mean_curvature_normal(V, F, d0.transpose()*W*d0, vorAreas, Hn, H);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Mean curvature computation took " << (double)(duration.count())/1000.0 << " seconds to execute." << std::endl;

    // Compute principal curvatures and directions (new in Section 4.1)
    start = std::chrono::high_resolution_clock::now();
    compute_principal_curvatures(V, F, k1, k2, d1, d2);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Principal curvature computation took " << (double)(duration.count())/1000.0 << " seconds to execute." << std::endl;

    // Compute Gaussian curvature = k1 * k2
    VectorXd K = k1.cwiseProduct(k2);

    // Compute mean curvature from principal curvatures H = (k1 + k2)/2
    VectorXd H_from_principal = (k1 + k2) * 0.5;

    // Compute shape index: SI = (2/π) * arctan((k1+k2)/(k1-k2))
    VectorXd shapeIndex = VectorXd::Zero(V.rows());
    for (int i = 0; i < V.rows(); i++) {
        if (std::abs(k1(i) - k2(i)) > 1e-10) {
            shapeIndex(i) = (2.0/M_PI) * std::atan((k1(i) + k2(i))/(k1(i) - k2(i)));
        }
    }

    // Compute curvedness: C = sqrt((k1^2 + k2^2)/2)
    VectorXd curvedness = (k1.array().square() + k2.array().square()).sqrt() * (1.0/std::sqrt(2.0));

    // Visualize Gaussian curvature
    psMesh->addVertexScalarQuantity("Gaussian Curvature (Angle Defect)", G.array()/vorAreas.array())->setEnabled(true);
    psMesh->addVertexScalarQuantity("Gaussian Curvature (k1*k2)", K);

    // Visualize Mean curvature
    psMesh->addVertexScalarQuantity("Mean Curvature (Laplacian)", H);
    psMesh->addVertexScalarQuantity("Mean Curvature (Principal)", H_from_principal);

    // Visualize Mean curvature normal
    psMesh->addVertexVectorQuantity("Mean Curvature Normal", Hn);

    // Visualize Principal curvatures as scalars
    psMesh->addVertexScalarQuantity("Maximum Principal Curvature (k1)", k1);
    psMesh->addVertexScalarQuantity("Minimum Principal Curvature (k2)", k2);

    // Visualize Shape index and curvedness
    psMesh->addVertexScalarQuantity("Shape Index", shapeIndex)
          ->setMapRange({-1.0, 1.0})
          ->setColorMap("coolwarm");
    psMesh->addVertexScalarQuantity("Curvedness", curvedness);

    // Visualize Principal directions as vector fields
    psMesh->addVertexVectorQuantity("Maximum Principal Direction", d1)
          ->setVectorColor({1.0, 0.0, 0.0});
    psMesh->addVertexVectorQuantity("Minimum Principal Direction", d2)
          ->setVectorColor({0.0, 0.0, 1.0});

    // Visualize curvature regions
    psMesh->addVertexScalarQuantity("Gaussian Regions", G.unaryExpr([](double x) { return (x > 0) - (x < 0); }));

    polyscope::show();

    return 0;
}