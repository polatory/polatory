// See LICENCE file at project root
//
#ifndef FPOINT_HPP
#define FPOINT_HPP

#include <array>
#include <iterator>
#include <ostream>
#include <istream>

#include "FMath.hpp"
#include "FConstFuncs.hpp"

/** N-dimentional coordinates
 *
 * \author Berenger Bramas <berenger.bramas@inria.fr>, Quentin Khan <quentin.khan@inria.fr>
 *
 * A fixed size array that represents coordinates in space. This class adds
 * a few convenience operations such as addition, scalar multiplication and
 * division and formated stream output.
 *
 * \tparam _Real The floating number type
 * \tparam Dim The space dimension
 **/
template<typename _Real, std::size_t _Dim = 3>
class FPoint : public std::array<_Real, _Dim> {
public:
    /// Floating number type
    using FReal = _Real;
    /// Space dimension count
    constexpr static const std::size_t Dim = _Dim;

private:

    /// Type used in SFINAE to authorize arithmetic types only in template parameters
    template<class T>
    using must_be_arithmetic = typename std::enable_if<std::is_arithmetic<T>::value>::type*;
    /// Type used in SFINAE to authorize floating point types only in template parameters
    template<class T>
    using must_be_floating = typename std::enable_if<std::is_floating_point<T>::value>::type*;
    /// Type used in SFINAE to authorize integral types only in template parameters
    template<class T>
    using must_be_integral = typename std::enable_if<std::is_integral<T>::value>::type*;


    /** Recursive template for constructor */
    template<typename A = FReal, typename... Args>
    void _fill_data(const A& arg, const Args... args) {
        this->data()[Dim-sizeof...(args)-1] = FReal(arg);
        _fill_data(args...);
    }

    /** Recursive template end condition for constructor */
    template<typename A = FReal>
    void _fill_data(const A& arg) {
        this->data()[Dim-1] = FReal(arg);
    }

public:

    /** Default constructor */
    FPoint() = default;
    /** Copy constructor */
    FPoint(const FPoint&) = default;

    /** Copy constructor from other point type */
    template<typename A, must_be_arithmetic<A> = nullptr>
    FPoint(const FPoint<A, Dim>& other) {
        for(std::size_t i = 0; i < Dim; ++i) {
            this->data()[i] = other.data()[i];
        }
    }

    /** Constructor from array */
    FPoint(const FReal array_[Dim]) {
        for(std::size_t i = 0; i < Dim; ++i) {
            this->data()[i] = array_[i];
        }
    }

    /** Constructor from args */
    template<typename A = FReal, typename... Args>
    FPoint(const FReal& arg, const Args... args) {
        static_assert(sizeof...(args)+1 == Dim, "FPoint argument list isn't the right length.");
        _fill_data(arg, args...);
    }

    /** Additive contructor, same as FPoint(other + add_value) */
    FPoint(const FPoint& other, const FReal add_value) {
        for(std::size_t i = 0; i < Dim; ++i) {
            this->data()[i] = other.data()[i] + add_value;
        }
    }

    /** Copies #Dim values from an iterable container
     *
     * \param other Container that defines begin() and end() methods.
     */
    template<class T>
    void copy(const T& other) {

        auto other_it = other.begin();
        auto this_it  = this->begin();
        for(std::size_t i = 0; i < Dim; ++i, ++this_it, ++other_it) {
            *this_it = *other_it;
        }
    }

    /** Assignment operator
     *
     * \param other A FPoint object.
     */
    template<typename T, must_be_arithmetic<T> = nullptr>
    FPoint<FReal, Dim>& operator=(const FPoint<T, Dim>& other) {
        this->copy(other);
        return *this;
    }


    /** Sets the point value */
    template<typename A = FReal, typename... Args>
    void setPosition(const FReal& X, const Args... YandZ) {
        static_assert(sizeof...(YandZ)+1 == Dim, "FPoint argument list isn't the right length.");
        _fill_data(X, YandZ...);
    }

    /** \brief Get x
     * \return this->data()[0]
     */
    FReal getX() const{
        return this->data()[0];
    }


    /** \brief Get y
     * \return this->data()[1]
     */
    FReal getY() const{
        return this->data()[1];
    }


    /** \brief Get z
     * \return this->data()[2]
     */
    FReal getZ() const{
        return this->data()[2];
    }


    /** \brief Set x
     * \param the new x
     */
    void setX(const FReal inX){
        this->data()[0] = inX;
    }


    /** \brief Set y
     * \param the new y
     */
    void setY(const FReal inY){
        this->data()[1] = inY;
    }


    /** \brief Set z
     * \param the new z
     */
    void setZ(const FReal inZ){
        this->data()[2] = inZ;
    }


    /** \brief Add to the x-dimension the inX value
     * \param  inXthe increment in x
     */
    void incX(const FReal inX){
        this->data()[0] += inX;
    }


    /** \brief Add to the y-dimension the inY value
     * \param  in<<<<<<y the increment in y
     */
    void incY(const FReal inY){
        this->data()[1] += inY;
    }


    /** \brief Add to z-dimension the inZ value
     * \param inZ the increment in z
     */
    void incZ(const FReal inZ){
        this->data()[2] += inZ;
    }

    /** \brief Get a pointer on the coordinate of FPoint<FReal>
     * \return the data value array
     */
    FReal * getDataValue(){
        return this->data() ;
    }

    /** \brief Get a pointer on the coordinate of FPoint<FReal>
     * \return the data value array
     */
    const FReal *  getDataValue()  const{
        return this->data() ;
    }

    /** \brief Compute the distance to the origin
     * \return the norm of the FPoint
     */
    FReal norm() const {
        return FMath::Sqrt(norm2()) ;
    }

    /** \brief Compute the distance to the origin
     * \return the square norm of the FPoint
     */
    FReal norm2() const {
        FReal square_sum = 0;
        for(std::size_t i = 0; i < Dim; ++i) {
            square_sum += this->data()[i]*this->data()[i];
        }
        return square_sum;
    }


    /** Addition assignment operator */
    template<class T, must_be_arithmetic<T> = nullptr>
    FPoint& operator +=(const FPoint<T, Dim>& other) {
        auto other_it = other.begin();
        auto this_it  = this->begin();
        for(std::size_t i = 0; i < Dim; ++i, ++this_it, ++other_it) {
            *this_it += *other_it;
        }

        return *this;
    }

    /** Addition operator */
    template<class T, must_be_arithmetic<T> = nullptr>
    friend FPoint<FReal, Dim> operator+(FPoint<FReal, Dim> lhs, const FPoint<T, Dim>& rhs) {
        lhs += rhs;
        return lhs;
    }

    /** Scalar assignment addition */
    template<class T, must_be_arithmetic<T> = nullptr>
    FPoint& operator+=(const T& val) {
        for(std::size_t i = 0; i < Dim; ++i) {
            this->data()[i] += val;
        }
        return *this;
    }

    /** Scalar addition */
    template<class T, must_be_arithmetic<T> = nullptr>
    friend FPoint<FReal, Dim> operator+(FPoint<FReal, Dim> lhs, const T& val) {
        lhs += val;
        return lhs;
    }

    /** Subtraction assignment operator */
    template<class T, must_be_arithmetic<T> = nullptr>
    FPoint& operator -=(const FPoint<T, Dim>& other) {
        auto other_it = other.begin();
        auto this_it  = this->begin();
        for(std::size_t i = 0; i < Dim; i++, ++this_it, ++other_it) {
            *this_it -= *other_it;
        }
        return *this;
    }

    /** Subtraction operator */
    template<class T, must_be_arithmetic<T> = nullptr>
    friend FPoint<FReal, Dim> operator-(FPoint<FReal, Dim> lhs, const FPoint<T, Dim>& rhs) {
        lhs -= rhs;
        return lhs;
    }

    /** Scalar subtraction assignment */
    template<class T, must_be_arithmetic<T> = nullptr>
    FPoint& operator-=(const T& val) {
        for(std::size_t i = 0; i < Dim; ++i) {
            this->data()[i] -= val;
        }
        return *this;
    }

    /** Scalar subtraction */
    template<class T, must_be_arithmetic<T> = nullptr>
    friend FPoint<FReal, Dim> operator-(FPoint<FReal, Dim> lhs, const T& rhs) {
        lhs -= rhs;
        return lhs;
    }


    /** Right scalar multiplication assignment operator */
    template<class T, must_be_arithmetic<T> = nullptr>
    FPoint<FReal, Dim>& operator *=(const T& val) {
        for(std::size_t i = 0; i < Dim; ++i) {
            this->data()[i] *= val;
        }
        return *this;
    }

    /** Right scalar multiplication operator */
    template<class T, must_be_arithmetic<T> = nullptr>
    friend FPoint<FReal, Dim> operator*(FPoint<FReal, Dim> lhs, const T& val) {
        lhs *= val;
        return lhs;
    }

    /** Left scalar multiplication operator */
    template<class T, must_be_arithmetic<T> = nullptr>
    friend FPoint<FReal, Dim> operator*(const T& val, FPoint<FReal, Dim> rhs) {
        rhs *= val;
        return rhs;
    }

    /** Data to data division assignment */
    template<class T, must_be_arithmetic<T> = nullptr>
    FPoint<FReal, Dim>& operator /=(const FPoint<T, Dim>& other) {
        for(std::size_t i = 0; i < Dim; ++i) {
            this->data()[i] *= other.data()[i];
        }
        return *this;
    }

    /** Data to data division operator */
    template<class T, must_be_arithmetic<T> = nullptr>
    friend FPoint<FReal, Dim> operator/(FPoint<FReal, Dim> lhs, const FPoint<FReal, Dim>& rhs) {
        lhs /= rhs;
        return lhs;
    }

    /** Right scalar division assignment operator */
    template<class T, must_be_arithmetic<T> = nullptr>
    FPoint<FReal, Dim>& operator /=(const T& val) {
        auto this_it  = this->begin();
        for(std::size_t i = 0; i < Dim; i++, ++this_it) {
            *this_it /= val;
        }

        return *this;
    }

    /** Right scalar division operator */
    template<class T, must_be_arithmetic<T> = nullptr>
    friend FPoint<FReal, Dim> operator/(FPoint<FReal, Dim> lhs, const T& val) {
        lhs /= val;
        return lhs;
    }

    /** Equality test operator */
    template<class T, must_be_integral<T> = nullptr>
    friend bool operator==(const FPoint<FReal, Dim>& lhs, const FPoint<T, Dim>& rhs) {
        auto lhs_it = lhs.begin(), rhs_it  = rhs.begin();
        for(std::size_t i = 0; i < Dim; i++, ++lhs_it, ++rhs_it) {
            if( *lhs_it != *rhs_it) {
                return false;
            }
        }
        return true;
    }

    /** Equality test operator */
    template<class T, must_be_floating<T> = nullptr>
    friend bool operator==(const FPoint<FReal, Dim>& lhs, const FPoint<T, Dim>& rhs) {
        auto lhs_it = lhs.begin(), rhs_it  = rhs.begin();
        for(std::size_t i = 0; i < Dim; i++, ++lhs_it, ++rhs_it) {
            if(! Ffeq(*lhs_it, *rhs_it)) {
                return false;
            }
        }
        return true;
    }

    /** Non equality test operator */
    template<class T, must_be_arithmetic<T> = nullptr>
    friend bool operator!=(const FPoint<FReal, Dim>& lhs, const FPoint<T, Dim>& rhs) {
        return ! (lhs == rhs);
    }

    /** Formated output stream operator */
    friend std::ostream& operator<<(std::ostream& os, const FPoint<FReal, Dim>& pos) {
        os << "[";
        for(auto it = pos.begin(); it != pos.end()-1; it++)
            os << *it << ", ";
        os << pos.back() << "]";
        return os;
    }

    /** Formated input stream operator */
    friend std::istream& operator>>(std::istream& is, FPoint<FReal, Dim>& pos) {
        char c;
        is >> c; // opening '['
        for(std::size_t i = 0; i + 1 < Dim; ++i) {
            is >> pos.data()[i];
            is >> c; // get coma ','
        }
        is >> pos.data()[Dim-1];
        is >> c; // closing ']'
        return is;
    }

    /** \brief Save current object */
    template <class BufferWriterClass>
    void save(BufferWriterClass& buffer) const {
        for(std::size_t i = 0; i < Dim; ++i) {
            buffer << this->data()[i];
        }
    }

    /** \brief Retrieve current object */
    template <class BufferReaderClass>
    void restore(BufferReaderClass& buffer) {
        for(std::size_t i = 0; i < Dim; ++i) {
            buffer >> this->data()[i];
        }
    }

};


#endif
