#ifndef __SERVER_H__
#define __SERVER_H__

#include "omnetpp.h"
#include "Job_m.h"

using namespace omnetpp;

// A simple Server
class Server : public cSimpleModule {
	private:
        //Signals
        simsignal_t iarrivalTimeSignal;
        simtime_t lastArrival;
        simsignal_t serviceTimeSignal;
        simsignal_t queueLengthSignal;
        simsignal_t nCustomersSignal;
        simsignal_t queueingTimeSignal;
        simsignal_t waitingTimeSignal;

        //Parameters
        int buffer;

        //Messages and Queue
        cMessage *endOfServiceMessage;
        Job *currentJob;
        cQueue queue;

        //Histograms
        simtime_t lastQueueLengthChange;
        simtime_t lastNCustomersChange;
        double lastQueueLength;
        double lastNCustomers;
        cAbstractHistogram *queueLengthHist;
        cAbstractHistogram *nCustomersHist;

	protected:
		virtual void initialize();
		virtual void handleMessage(cMessage *msg);
		virtual void finish();

	//Helper
        virtual void arrival(Job *job);
        virtual simtime_t startService(Job *job);
        virtual void endService(Job *job);
        virtual Job *dequeue();
        virtual void enqueue(Job *job);

	public:
		Server();
		virtual ~Server();
};

#endif
