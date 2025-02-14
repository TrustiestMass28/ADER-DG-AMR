#ifndef NUMERICALMETHOD_H
#define NUMERICALMETHOD_H

#include <memory>

//construct derived:AmrDG
//  cosntructr base: NumericalMethod
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

#endif 


