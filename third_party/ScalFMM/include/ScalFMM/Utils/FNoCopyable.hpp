// See LICENCE file at project root
#ifndef FNOCOPYABLE_HPP
#define FNOCOPYABLE_HPP

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* This class has to be inherited to forbid copy
* @todo use C++0x ?
*/
class FNoCopyable {
private:
        /** Forbiden copy constructor */
        FNoCopyable(const FNoCopyable&) = delete;
        /** Forbiden copy operator */
        FNoCopyable& operator=(const FNoCopyable&) = delete;
protected:
        /** Empty constructor */
        FNoCopyable(){}
};

/**
* @author Berenger Bramas (berenger.bramas@inria.fr)
* This class has to be inherited to forbid assignement
*/
class FNoAssignement {
private:
        /** Forbiden copy operator */
        FNoAssignement& operator=(const FNoAssignement&) = delete;
protected:
        /** Empty constructor */
        FNoAssignement(){}
};

#endif // FNOCOPYABLE_HPP
