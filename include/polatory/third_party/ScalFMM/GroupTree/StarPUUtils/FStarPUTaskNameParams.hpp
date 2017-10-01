#ifndef FSTARPUTASKNAMEPARAMS_HPP
#define FSTARPUTASKNAMEPARAMS_HPP

#include "../../Utils/FGlobal.hpp"

#include <list>
#include <cstring>
#include <cstdio>

/**
 * This class creates task name for starpu
 * it is used for simgrid (to pass task parameters)
 */
class FStarPUTaskNameParams{
protected:
    std::list<const char*> names;
    FILE* fout;
    int taskid;

public:
    FStarPUTaskNameParams() : fout(nullptr), taskid(0){
        const char* fname = getenv("SCALFMM_SIMGRIDOUT")?getenv("SCALFMM_SIMGRIDOUT"):"/tmp/scalfmm.out";
        fout = fopen(fname, "w");
        std::cout << "output task name in " << fname << "\n";
    }

    ~FStarPUTaskNameParams(){
        fclose(fout);
        clear();
    }

    void clear(){
        while(names.size()){
            delete[] names.front();
            names.pop_front();
        }
    }

    template <typename ... Params>
    const char* print(const char key[], const char format[], Params... args ){
        const size_t length = 512;
        char* name = new char[length+1];
        snprintf(name, length, "%s_%d", key, taskid++);
        name[length] = '\0';
        names.push_back(name);

        fprintf(fout, "%s, %d, ", key, taskid);
        fprintf(fout, format, args...);

        return name;
    }

    const char* add(const char key[], const char* strToCpy){
        const size_t length = 512;
        char* name = new char[length+1];
        snprintf(name, length, "%s_%d", key, taskid++);
        name[length] = '\0';
        names.push_back(name);

        fprintf(fout, "%s=", name);
        fprintf(fout, strToCpy);

        return name;
    }
};

#endif // FSTARPUTASKNAMEPARAMS_HPP
