#ifndef __SOURCE_H__
#define __SOURCE_H__

#include "omnetpp.h"
#include "Job_m.h"

using namespace omnetpp;

class Source : public cSimpleModule {
    private:
        cMessage *sendMessageEvent;
    protected:
        virtual void initialize();
        virtual void handleMessage(cMessage *msg);
	public:
		Source();
		virtual ~Source();
};

#endif
