/*NDPLUGINSSCPIMEGA_H
 *
 * Authors: Marco A. B Montevechi Filho, Henrique F. Simoes, Eduardo X. S. Miqueles
 *
 */
#ifndef NDPLUGINSSCPIMEGA_H
#define NDPLUGINSSCPIMEGA_H

#include <memory>
#include <vector>

#include <epicsTypes.h>
#include <shareLib.h>

#include "NDPluginDriver.h"

/* Param definitions */

class epicsShareClass NDPluginSSCPimega : public NDPluginDriver {
public:
    NDPluginSSCPimega(const char *portName, int queueSize, int blockingCallbacks,
                 const char *NDArrayPort, int NDArrayAddr,
                 int maxBuffers, size_t maxMemory,
                 int priority, int stackSize);
    virtual ~NDPluginSSCPimega();

    virtual void processCallbacks(NDArray *pArray);
    virtual asynStatus writeFloat64(asynUser *pasynUser, epicsFloat64 value);
    virtual asynStatus writeInt32(asynUser *pasynUser, epicsInt32 value);

    asynStatus loadMatrix();
    NDArrayInfo lastinfo;

protected:

    int blockSize;
    int blockSizeVal;
    #define FIRST_NDPLUGIN_SSC_PIMEGA_PARAM blockSize
    int pimegaModel;
    int pimegaModelVal;

    int loadMatrixes;
    int matrixFilePath;

    char matrixFileName[256];

};

#define SSCPimegaBlockSizeString      "BLOCK_SIZE"
#define SSCPimegaModelString          "PIMEGA_MODEL"
#define SSCPimegaLoadMatrixString     "LOAD_MATRIX"
#define SSCPimegaFilePath             "MATRIX_FILE_PATH"

#endif // NDPLUGINSSCPIMEGA_H
