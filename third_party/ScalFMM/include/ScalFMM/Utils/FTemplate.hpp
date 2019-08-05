// See LICENCE file at project root
#ifndef FTEMPLATE_HPP
#define FTEMPLATE_HPP

///////////////////////////////////////////////////////////////////////////////////////
/// This file provide useful method to work with template.
/// It provide solution in order to build several methods
/// and run them accordingly to a given condition.
/// We recommand to look at the testTemplateExample.cpp in order
/// to see the usage.
///////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////////////
/// FForAll : Compile all and exec all
///////////////////////////////////////////////////////////////////////////////////////

#include <functional>

namespace FForAll{

template <class IterType, const IterType CurrentIter, const IterType iterTo, const IterType IterStep,
          class Func, bool IsNotOver, typename... Args>
struct Evaluator{
    static void Run(Args... args){
        Func::template For<CurrentIter>(args...);
        Evaluator<IterType, CurrentIter+IterStep, iterTo, IterStep, Func, (CurrentIter+IterStep < iterTo), Args...>::Run(args...);
    }
};

template <class IterType, const IterType CurrentIter, const IterType iterTo, const IterType IterStep,
          class Func, typename... Args>
struct Evaluator< IterType, CurrentIter, iterTo, IterStep, Func, false, Args...>{
    static void Run(Args... args){
    }
};

template <class IterType, const IterType IterFrom, const IterType iterTo, const IterType IterStep,
          class Func, typename... Args>
void For(Args... args){
    Evaluator<IterType, IterFrom, iterTo, IterStep, Func, (IterFrom<iterTo), Args...>::Run(args...);
}

}


///////////////////////////////////////////////////////////////////////////////////////
/// FForAll : Compile all and exec all
///////////////////////////////////////////////////////////////////////////////////////

#include <functional>

namespace FForAllWithInc{

template <class IterType, const IterType CurrentIter, const IterType iterTo, template <IterType> class ClassStep,
          class Func, bool IsNotOver, typename... Args>
struct Evaluator{
    static void Run(Args... args){
        Func::template For<CurrentIter>(args...);
        Evaluator<IterType, ClassStep<CurrentIter>::NextValue, iterTo, ClassStep, Func, (ClassStep<CurrentIter>::NextValue < iterTo), Args...>::Run(args...);
    }
};

template <class IterType, const IterType CurrentIter, const IterType iterTo, template <IterType> class ClassStep,
          class Func, typename... Args>
struct Evaluator< IterType, CurrentIter, iterTo, ClassStep, Func, false, Args...>{
    static void Run(Args... args){
    }
};

template <class IterType, const IterType IterFrom, const IterType iterTo, template <IterType> class ClassStep,
          class Func, typename... Args>
void For(Args... args){
    Evaluator<IterType, IterFrom, iterTo, ClassStep, Func, (IterFrom<iterTo), Args...>::Run(args...);
}

}

///////////////////////////////////////////////////////////////////////////////////////
/// FForAll : Compile all and exec all
///////////////////////////////////////////////////////////////////////////////////////

#include <functional>

namespace FForAllThis{

template <class IterType, const IterType CurrentIter, const IterType iterTo, const IterType IterStep,
          class Func, bool IsNotOver, typename... Args>
struct Evaluator{
    static void Run(Func* object, Args... args){
        object->Func::template For<CurrentIter>(args...);
        Evaluator<IterType, CurrentIter+IterStep, iterTo, IterStep, Func, (CurrentIter+IterStep < iterTo), Args...>::Run(object, args...);
    }
};

template <class IterType, const IterType CurrentIter, const IterType iterTo, const IterType IterStep,
          class Func, typename... Args>
struct Evaluator< IterType, CurrentIter, iterTo, IterStep, Func, false, Args...>{
    static void Run(Func* object, Args... args){
    }
};

template <class IterType, const IterType IterFrom, const IterType iterTo, const IterType IterStep,
          class Func, typename... Args>
void For(Func* object, Args... args){
    Evaluator<IterType, IterFrom, iterTo, IterStep, Func, (IterFrom<iterTo), Args...>::Run(object, args...);
}

}



///////////////////////////////////////////////////////////////////////////////////////
/// FForAll : Compile all and exec all
///////////////////////////////////////////////////////////////////////////////////////

#include <functional>

namespace FForAllThisWithInc{

template <class IterType, const IterType CurrentIter, const IterType iterTo, template <IterType> class ClassStep,
          class Func, bool IsNotOver, typename... Args>
struct Evaluator{
    static void Run(Func* object, Args... args){
        object->Func::template For<CurrentIter>(args...);
        Evaluator<IterType, ClassStep<CurrentIter>::NextValue, iterTo, ClassStep, Func, (ClassStep<CurrentIter>::NextValue < iterTo), Args...>::Run(object, args...);
    }
};

template <class IterType, const IterType CurrentIter, const IterType iterTo, template <IterType> class ClassStep,
          class Func, typename... Args>
struct Evaluator< IterType, CurrentIter, iterTo, ClassStep, Func, false, Args...>{
    static void Run(Func* object, Args... args){
    }
};

template <class IterType, const IterType IterFrom, const IterType iterTo, template <IterType> class ClassStep,
          class Func, typename... Args>
void For(Func* object, Args... args){
    Evaluator<IterType, IterFrom, iterTo, ClassStep, Func, (IterFrom<iterTo), Args...>::Run(object, args...);
}

}

///////////////////////////////////////////////////////////////////////////////////////
/// FRunIf : Compile all and exec only one (if the template variable is equal to
/// the first variable)
///////////////////////////////////////////////////////////////////////////////////////

#include <iostream>

namespace FRunIf{

template <class IterType, const IterType CurrentIter, const IterType iterTo, const IterType IterStep,
          class Func, bool IsNotOver, typename... Args>
struct Evaluator{
    static void Run(IterType value, Args... args){
        if(CurrentIter == value){
            Func::template Run<CurrentIter>(args...);
        }
        else{
            Evaluator<IterType, CurrentIter+IterStep, iterTo, IterStep, Func, (CurrentIter+IterStep < iterTo), Args...>::Run(value, args...);
        }
    }
};

template <class IterType, const IterType CurrentIter, const IterType iterTo, const IterType IterStep,
          class Func, typename... Args>
struct Evaluator< IterType, CurrentIter, iterTo, IterStep, Func, false, Args...>{
    static void Run(IterType value, Args... args){
        std::cout << __FUNCTION__ << " no matching value found\n";
    }
};

template <class IterType, const IterType IterFrom, const IterType iterTo, const IterType IterStep,
          class Func, typename... Args>
void Run(IterType value, Args... args){
    Evaluator<IterType, IterFrom, iterTo, IterStep, Func, (IterFrom<iterTo), Args...>::Run(value, args...);
}

}


///////////////////////////////////////////////////////////////////////////////////////
/// FRunIf : Compile all and exec only one (if the template variable is equal to
/// the first variable)
///////////////////////////////////////////////////////////////////////////////////////

namespace FRunIfThis{

template <class IterType, const IterType CurrentIter, const IterType iterTo, const IterType IterStep,
          class Func, bool IsNotOver, typename... Args>
struct Evaluator{
    static void Run(Func* object, IterType value, Args... args){
        if(CurrentIter == value){
            object->Func::template Run<CurrentIter>(args...);
        }
        else{
            Evaluator<IterType, CurrentIter+IterStep, iterTo, IterStep, Func, (CurrentIter+IterStep < iterTo), Args...>::Run(object, value, args...);
        }
    }
};

template <class IterType, const IterType CurrentIter, const IterType iterTo, const IterType IterStep,
          class Func, typename... Args>
struct Evaluator< IterType, CurrentIter, iterTo, IterStep, Func, false, Args...>{
    static void Run(Func* object, IterType value, Args... args){
        std::cout << __FUNCTION__ << " no matching value found\n";
    }
};

template <class IterType, const IterType IterFrom, const IterType iterTo, const IterType IterStep,
          class Func, typename... Args>
void Run(Func* object, IterType value, Args... args){
    Evaluator<IterType, IterFrom, iterTo, IterStep, Func, (IterFrom<iterTo), Args...>::Run(object, value, args...);
}

}


///////////////////////////////////////////////////////////////////////////////////////
/// FRunIfFunctional : Compile all and exec only those whose respect a condition
///////////////////////////////////////////////////////////////////////////////////////

namespace FRunIfFunctional{

template <class IterType, const IterType CurrentIter, const IterType iterTo, const IterType IterStep,
          class Func, bool IsNotOver, typename... Args>
struct Evaluator{
    static void Run(std::function<bool(IterType)> test, Args... args){
        if(test(CurrentIter)){
            Func::template Run<CurrentIter>(args...);
        }
        Evaluator<IterType, CurrentIter+IterStep, iterTo, IterStep, Func, (CurrentIter+IterStep < iterTo), Args...>::Run(test, args...);
    }
};

template <class IterType, const IterType CurrentIter, const IterType iterTo, const IterType IterStep,
          class Func, typename... Args>
struct Evaluator< IterType, CurrentIter, iterTo, IterStep, Func, false, Args...>{
    static void Run(std::function<bool(IterType)> test, Args... args){
    }
};

template <class IterType, const IterType IterFrom, const IterType iterTo, const IterType IterStep,
          class Func, typename... Args>
void Run(std::function<bool(IterType)> test,  Args... args){
    Evaluator<IterType, IterFrom, iterTo, IterStep, Func, (IterFrom<iterTo), Args...>::Run(test, args...);
}

}

#endif // FTEMPLATE_HPP
