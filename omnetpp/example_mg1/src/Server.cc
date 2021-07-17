#include "Server.h"

Define_Module(Server);

Server::Server() {
    //Messages and Queue
	endOfServiceMessage = nullptr;
	currentJob = nullptr;

	//Histograms
	queueLengthHist = nullptr;
	nCustomersHist = nullptr;
}

Server::~Server() {
    //Messages and Queue
    cancelAndDelete(endOfServiceMessage);
    delete currentJob;

    //Histograms
    delete queueLengthHist;
    delete nCustomersHist;
}

void Server::initialize() {
    //Signals
    iarrivalTimeSignal  = registerSignal("iarrivalTime");
    lastArrival = 0.0;
    serviceTimeSignal = registerSignal("serviceTime");
    queueLengthSignal  = registerSignal("queueLength");
    nCustomersSignal  = registerSignal("nCustomers");
    queueingTimeSignal = registerSignal("queueingTime");
    waitingTimeSignal  = registerSignal("waitingTime");
//    emit(queueLengthSignal, 0);
//    emit(nCustomersSignal, 0);

    //Parameters
    buffer = long(par("buffer"));

    //Messages and Queue
    endOfServiceMessage = new cMessage("EoS");
    queue.setName("queue");

    //Histogram
    lastQueueLengthChange = 0.0;
    lastNCustomersChange = 0.0;
    lastQueueLength = 0.0;
    lastNCustomers = 0.0;
    queueLengthHist = new cHistogram("queueLengthHist", true);
    nCustomersHist = new cHistogram("nCustomersHist", true);
}

void Server::handleMessage(cMessage *msg) {
    if (msg==endOfServiceMessage) {
		// Job has finished service
        endService(currentJob);
        if (!queue.isEmpty()) {
            // start next waiting job
            currentJob = dequeue();
            simtime_t serviceTime = startService(currentJob);
            scheduleAt(simTime()+serviceTime, endOfServiceMessage);
        } else {
        	// no job waiting in queue
        	currentJob = nullptr;
        }
    } else {
		// arrival of new job
        Job *job = check_and_cast<Job *>(msg);
        arrival(job);
    	if (currentJob == nullptr) {
    		// no job in service => start service for newly arrived job
    		currentJob = job;
            simtime_t serviceTime = startService(currentJob);
			scheduleAt(simTime()+serviceTime, endOfServiceMessage);
    	} else {
    		// server is busy => put job in queue if possible
			if (buffer >= 0 && queue.getLength() >= buffer) {
                // queue is full => job is lost
                delete job;
			} else {
			    enqueue(job);
			}
    	}
    }
}

void Server::finish() {
    if (lastQueueLengthChange < simTime()) {
        // queue length statistics
        double queueLength = queue.getLength();
        emit(queueLengthSignal, queueLength);
        // queue length histogram
        double value = lastQueueLength;
        double weight = simTime().dbl() - lastQueueLengthChange.dbl();
        queueLengthHist->collectWeighted(value, weight);
        lastQueueLength = queueLength;
        lastQueueLengthChange = simTime();
    }
    if (lastNCustomersChange < simTime()) {
        // customers statistics
        double custInService = (currentJob == nullptr) ? 0 : 1;
        double nCustomers = queue.getLength() + custInService;
        emit(nCustomersSignal, nCustomers);
        // customers histogram
        double value = lastNCustomers;
        double weight = simTime().dbl() - lastNCustomersChange.dbl();
        nCustomersHist->collectWeighted(value, weight);
        lastNCustomers = nCustomers;
        lastNCustomersChange = simTime();
    }
    queueLengthHist->record();
    nCustomersHist->record();
}

void Server::arrival(Job *job) {
    EV << "Arrival of " << job->getName() << endl;
    simtime_t iarrivalTime = simTime() - lastArrival;
    emit(iarrivalTimeSignal, iarrivalTime);
    lastArrival = simTime();
    // customers statistics
    double custInService = (currentJob == nullptr) ? 0 : 1;
    double nCustomers = queue.getLength() + custInService + 1;
    emit(nCustomersSignal, nCustomers);
    // customers histogram
    double value = lastNCustomers;
    double weight = simTime().dbl() - lastNCustomersChange.dbl();
    nCustomersHist->collectWeighted(value, weight);
    lastNCustomers = nCustomers;
    lastNCustomersChange = simTime();
    //
    job->setTimestamp();
}

simtime_t Server::startService(Job *job) {
    EV << "Starting service of " << job->getName() << endl;
    // queueing time statistics
    simtime_t queueingTime = simTime() - job->getTimestamp();
    job->setQueueingTime(queueingTime);
    emit(queueingTimeSignal, queueingTime);
    // return next serviceTime
    job->setTimestamp();
    simtime_t serviceTime;
    serviceTime = par("serviceTime");
    return serviceTime;
}

void Server::endService(Job *job) {
    EV << "Completed service of " << job->getName() << endl;
    // customers statistics
    double nCustomers = queue.getLength();
    emit(nCustomersSignal, nCustomers);
    // customers histogram
    double value = lastNCustomers;
    double weight = simTime().dbl() - lastNCustomersChange.dbl();
    nCustomersHist->collectWeighted(value, weight);
    lastNCustomers = nCustomers;
    lastNCustomersChange = simTime();
    // waiting time statistics
    simtime_t serviceTime = simTime() - job->getTimestamp();
    simtime_t waitingTime = job->getQueueingTime() + serviceTime;
    job->setServiceTime(serviceTime);
    job->setWaitingTime(waitingTime);
    emit(serviceTimeSignal, serviceTime);
    emit(waitingTimeSignal, waitingTime);
    // send service to sink
    job->setTimestamp();
    send(job, "out");
}

Job *Server::dequeue() {
    EV << "Dequeued next service" << endl;
    // dequeue next job
    Job *job = (Job *)queue.pop();
    // queue length statistics
    double queueLength = queue.getLength();
    emit(queueLengthSignal, queueLength);
    // queue length histogram
    double value = lastQueueLength;
    double weight = simTime().dbl() - lastQueueLengthChange.dbl();
    queueLengthHist->collectWeighted(value, weight);
    lastQueueLength = queueLength;
    lastQueueLengthChange = simTime();
    //
    return job;
}

void Server::enqueue(Job *job) {
    EV << "Enqueue service of " << job->getName() << endl;
    // enqueue job
    queue.insert(job);
    // queue length statistics
    double queueLength = queue.getLength();
    emit(queueLengthSignal, queueLength);
    // queue length histogram
    double value = lastQueueLength;
    double weight = simTime().dbl() - lastQueueLengthChange.dbl();
    queueLengthHist->collectWeighted(value, weight);
    lastQueueLength = queueLength;
    lastQueueLengthChange = simTime();
}

