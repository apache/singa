/**
 * @file model_save_load.cpp
 * Demonstrate model save/load via libpq
 */

#include <libpq-fe.h>
#include <iostream>
#include <onnx.pb.h>
#include <string>
#include <arpa/inet.h>
#include <filesystem>


void set_parallelism(PGconn *conn, const int parallelism) {
    const int32_t parallelism_be = htonl(parallelism);
    const char *paramValues[1] = {reinterpret_cast<const char *>(&parallelism_be)};
    constexpr int paramLengths[1] = {sizeof(int32_t)};
    constexpr int paramFormats[1] = {1};
    constexpr Oid paramTypes[1] = {23}; // OID for integer

    PGresult *res = PQexecParams(
        conn,
        "SELECT ns_set_parallelism($1)",
        1,
        paramTypes,
        paramValues,
        paramLengths,
        paramFormats,
        0
    );

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Failed to set parallelism: " << PQresultErrorMessage(res) << std::endl;
    } else {
        std::cout << "Set neurstore parallelism to " << parallelism << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }
    PQclear(res);
}

void save_models_from_folder(PGconn *conn, const std::string &folder_path, const float tolerance) {
    const char *paramValues[2];
    int paramLengths[2];
    int paramFormats[2];
    Oid paramTypes[2] = {25, 700};  // OID for text and float4

    paramValues[0] = folder_path.c_str();
    paramValues[1] = reinterpret_cast<const char *>(&tolerance);

    paramLengths[0] = folder_path.size();
    paramLengths[1] = sizeof(float);

    paramFormats[0] = 0;
    paramFormats[1] = 1;

    PGresult *res = PQexecParams(
        conn,
        "SELECT ns_save_models_from_folder($1, $2)",
        2,
        paramTypes,
        paramValues,
        paramLengths,
        paramFormats,
        0
    );

    if (PQresultStatus(res) != PGRES_TUPLES_OK) {
        std::cerr << "Failed to save models from folder: " << PQresultErrorMessage(res) << std::endl;
    } else {
        std::cout << "Saved all models from: " << folder_path << std::endl;
        std::cout << "--------------------------------" << std::endl;
    }
    PQclear(res);
}

void load_model(PGconn* conn, const int model_id, const std::string& udf, const bool return_serialized = false) {
    const int32_t model_id_be = htonl(model_id);
    const char* return_serialized_model = return_serialized ? "t" : "f";
    const char* paramValues[2] = { reinterpret_cast<const char*>(&model_id_be), return_serialized_model };
    constexpr int paramLengths[2] = { sizeof(int32_t), 1 };
    constexpr int paramFormats[2] = { 1, 0 };
    constexpr Oid paramTypes[2] = { 23, 16 };

    const std::string query = "SELECT " + udf + "($1, $2)";

    PGresult* res = PQexecParams(
        conn,
        query.c_str(),
        2,
        paramTypes,
        paramValues,
        paramLengths,
        paramFormats,
        1
    );

    if (PQresultStatus(res) != PGRES_TUPLES_OK || PQntuples(res) == 0) {
        std::cerr << "Failed to load model with " << udf << ": " << PQresultErrorMessage(res) << std::endl;
        PQclear(res);
        return;
    }

    if (return_serialized) {
        const char* data = PQgetvalue(res, 0, 0);
        const int size = PQgetlength(res, 0, 0);
        onnx::ModelProto model;
        if (!model.ParseFromArray(data, size)) {
            std::cerr << "Failed to parse model data." << std::endl;
            PQclear(res);
            return;
        }
    }

    std::cout << "Loaded model (id = " << model_id << ") using " << udf << std::endl;
    std::cout << "--------------------------------" << std::endl;
    PQclear(res);
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <conninfo> <tolerance> <model_folder>" << std::endl;
        return 1;
    }
    const std::string conninfo = argv[1];
    const float tolerance = std::stof(argv[2]);
    const std::string model_folder = argv[3];

    const std::string abs_model_folder = std::filesystem::absolute(model_folder).string();

    PGconn *conn = PQconnectdb(conninfo.c_str());
    if (PQstatus(conn) != CONNECTION_OK) {
        std::cerr << "Connection failed: " << PQerrorMessage(conn) << std::endl;
        return 1;
    }
    std::cout << "PostgreSQL connection established" << std::endl;
    std::cout << "conninfo: " << conninfo << std::endl;
    std::cout << "compression tolerance: " << tolerance << std::endl;
    std::cout << "--------------------------------" << std::endl;

    set_parallelism(conn, 1);
    save_models_from_folder(conn, abs_model_folder, tolerance);

    constexpr int model_id = 1;
    load_model(conn, model_id, "ns_load_model");
    load_model(conn, model_id, "ns_load_model_as_uint8_delta", true);

    PQfinish(conn);
    return 0;
}
