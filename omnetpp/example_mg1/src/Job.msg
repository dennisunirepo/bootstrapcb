message Job
{
    simtime_t queueingTime; // total time spent standing in queues
    simtime_t serviceTime;  // total time spent in servers
    simtime_t waitingTime;  // total time spent in servers
}


