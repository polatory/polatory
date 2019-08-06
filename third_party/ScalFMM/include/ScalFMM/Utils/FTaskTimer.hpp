#ifndef FTASKTIMER_HPP
#define FTASKTIMER_HPP

#include "FGlobal.hpp"
#include "FTic.hpp"
#include "FAssert.hpp"

#include "../Containers/FVector.hpp"

#include <unordered_set>


#ifdef SCALFMM_TIME_OMPTASKS
#define FTIME_TASKS(X) X;
#else
#define FTIME_TASKS(X)
#endif

/**
 * @brief The FTaskTimer class
 * you can find at the bottom of the file a way of loading a file generated by FTaskTimer
 */
class FTaskTimer{
protected:
    static const int MaxTextLength = 16;

    struct EventDescriptor{
        char text[MaxTextLength];
        double duration;
        double start;
        long long int eventId;
        int threadId;
    };

    struct ThreadData{
        FVector<EventDescriptor> events;
    };

    const int nbThreads;
    // We create array of ptr to avoid the modifcation of clause
    // memory by different threads
    ThreadData** threadEvents;
    double startingTime;
    double duration;

public:

    explicit FTaskTimer(const int inNbThreads)
            : nbThreads(inNbThreads), threadEvents(nullptr),
                startingTime(0) {
        FLOG( FLog::Controller << "\tFTaskTimer is used\n" );

        threadEvents = new ThreadData*[nbThreads];
        memset(threadEvents, 0, sizeof(threadEvents[0])*nbThreads);
    }

    void init(const int threadId){
        threadEvents[threadId] = new ThreadData;
    }

    ~FTaskTimer(){
        for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
            delete threadEvents[idxThread];
        }
        delete[] threadEvents;
    }

    void start(){
        FLOG( FLog::Controller << "\tFTaskTimer starts\n" );
        for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
            if(threadEvents[idxThread]) threadEvents[idxThread]->events.clear();
        }
        startingTime = FTic::GetTime();
    }

    void end(){
        FLOG( FLog::Controller << "\tFTaskTimer ends\n" );
        duration = FTic::GetTime() - startingTime;
    }

    void saveToDisk(const char inFilename[]) const {
        FLOG( FLog::Controller << "\tFTaskTimer saved to " << inFilename << "\n" );
        FILE* foutput = fopen(inFilename, "w");
        FAssert(foutput);

        fprintf(foutput, "ScalFMM Task Records\n");

        FSize totalEvents = 0;
        for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
            if(threadEvents[idxThread]){
                totalEvents += threadEvents[idxThread]->events.getSize();
            }
        }
        fprintf(foutput, "global{@duration=%e;@max threads=%d;@nb events=%lld}\n",
                duration, nbThreads, totalEvents);

        std::unordered_set<long long int> ensureUniqueness;
        ensureUniqueness.reserve(totalEvents);

        for(int idxThread = 0 ; idxThread < nbThreads ; ++idxThread){
            if(threadEvents[idxThread]){
                for(int idxEvent = 0 ; idxEvent < threadEvents[idxThread]->events.getSize() ; ++idxEvent){
                    const EventDescriptor& event = threadEvents[idxThread]->events[idxEvent];
                    fprintf(foutput, "event{@id=%lld;@thread=%d;@duration=%e;@start=%e;@text=%s}\n",
                            event.eventId, event.threadId, event.duration, event.start, event.text);
                    FAssertLF(ensureUniqueness.find(event.eventId) == ensureUniqueness.end());
                    ensureUniqueness.insert(event.eventId);
                }
            }
        }

        fclose(foutput);
    }

    class ScopeEvent{
    protected:
        const int threadId;
        const double eventStartingTime;
        const double measureStartingTime;
        ThreadData*const  myEvents;

        const long long int taskId;
        char taskText[MaxTextLength];

    public:
        ScopeEvent(const int inThreadId, FTaskTimer* eventsManager, const long long int inTaskId, const char inText[MaxTextLength])
            : threadId(inThreadId), eventStartingTime(FTic::GetTime()), measureStartingTime(eventsManager->startingTime),
              myEvents(eventsManager->threadEvents[inThreadId]),
              taskId(inTaskId){
            taskText[0] = '\0';
            strncpy(taskText, inText, MaxTextLength);
        }

        template <class FirstParameters, class ... Parameters>
        ScopeEvent(const int inThreadId, FTaskTimer* eventsManager, const long long int inTaskId, const char inTextFormat, FirstParameters fparam, Parameters ... params)
            : threadId(inThreadId), eventStartingTime(FTic::GetTime()), measureStartingTime(eventsManager->startingTime),
              myEvents(eventsManager->threadEvents[inThreadId]),
              taskId(inTaskId){
            snprintf(taskText, MaxTextLength, inTextFormat, fparam, params...);
        }

        ~ScopeEvent(){
            EventDescriptor event;
            event.duration = FTic::GetTime()-eventStartingTime;
            event.eventId  = taskId;
            event.threadId = threadId;
            event.start    = eventStartingTime-measureStartingTime;
            strncpy(event.text, taskText, MaxTextLength);
            myEvents->events.push(event);
        }
    };
};


/*
#include <cstdio>
#include <cassert>

static const int MaxTextLength = 16;

struct EventDescriptor{
    int threadId;
    char text[MaxTextLength];
    double duration;
    double start;
    long long int eventId;
};

int main(int argc, char** argv){
    assert(argc == 2);
    FILE* ftime = fopen(argv[1], "r");
    assert(ftime);

    fscanf(ftime,"%*[^\n]\n");

    double duration;
    int nbThreads;
    long long int nbEvents;
    // written by "global{@duration=%e;@max threads=%d;@nb events=%lld}\n"
    assert(fscanf(ftime,"global{@duration=%lf;@max threads=%d;@nb events=%lld}\n",
        &duration,&nbThreads,&nbEvents) == 3);
    printf("global{@duration=%e;@max threads=%d;@nb events=%lld}\n",
        duration,nbThreads,nbEvents);


    EventDescriptor* events = new EventDescriptor[nbEvents];
    for(int idxEvent = 0 ; idxEvent < nbEvents ; ++idxEvent){
        char line[1024];
        fgets(line, 1024, ftime);
        // written format "event{@id=%lld;@duration=%e;@start=%e;@text=%s}\n"
        assert(sscanf(line,"event{@id=%lld;@thread=%d;@duration=%lf;@start=%lf;@text=%[^}]s%*[^\n]\n",
            &events[idxEvent].eventId,&events[idxEvent].threadId,&events[idxEvent].duration,&events[idxEvent].start,events[idxEvent].text) == 5);
        printf("event{@id=%lld;@thread=%d;@duration=%lf;@start=%lf;@text=%s}\n",
                events[idxEvent].eventId,events[idxEvent].threadId,events[idxEvent].duration, events[idxEvent].start, events[idxEvent].text);
    }

    fclose(ftime);
    return 0;
}
  */

#endif // FTASKTIMER_HPP
