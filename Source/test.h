#ifndef TEST_H_
#define TEST_H_

#include "ModelEquation.h"

class Test : public ModelEquation
{
  public:

    Test(int a);

    ~Test() override {};
};

#endif TEST_H_