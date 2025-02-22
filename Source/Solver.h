#ifndef SOLVER_H
#define SOLVER_H

#include <iostream>
#include <memory>
#include <any>

using namespace amrex;


template <typename NumericalMethodType>
class Solver
{
    public: 

        Solver() = default;

        virtual ~Solver() = default;
        
        template <typename... Args>
        void settings(Args... args) {
            std::cout << "test" << std::endl;
            static_cast<NumericalMethodType*>(this)->settings(std::forward<Args>(args)...);
        }



        void setNumericalMethod(std::shared_ptr<NumericalMethodType> nm) {
            numerical_pde = nm;
        }

        // Getter methods
        std::shared_ptr<NumericalMethodType> getNumericalMethod() const {
            return numerical_pde;
        }
        
        void setOfstream(std::shared_ptr<std::ofstream> _ofs) {
            ofs = _ofs;
        }

        void init()
        {
            static_cast<NumericalMethodType*>(this)->init();
        }



    protected:

        std::shared_ptr<NumericalMethodType> numerical_pde; 

        std::shared_ptr<std::ofstream> ofs;
};

#endif 

