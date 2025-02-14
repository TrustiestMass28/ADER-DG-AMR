#ifndef NUMERICALMETHOD_H_
#define NUMERICALMETHOD_H_

#include <memory>

class ModelEquation;

class NumericalMethod
{
    public: 
        NumericalMethod() {};

        virtual ~NumericalMethod() = default;

        // Set reference to ModelEquation for communication
        void setModelEquation(std::shared_ptr<ModelEquation> me){
            model_pde = me;
        }

        void setOfstream(std::shared_ptr<std::ofstream> _ofs) {
            ofs = _ofs;
        }

    protected:

        std::shared_ptr<std::ofstream> ofs;

        std::shared_ptr<ModelEquation> model_pde;

};

#endif NUMERICALMETHOD_H_


