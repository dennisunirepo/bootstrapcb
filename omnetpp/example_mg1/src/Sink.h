#ifndef __SINK_H__
#define __SINK_H__

#include "omnetpp.h"

using namespace omnetpp;

class Sink : public cSimpleModule {
  protected:
    virtual void initialize();
    virtual void handleMessage(cMessage *msg);
};

#endif
