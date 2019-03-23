#include "range.hpp"
#include <iostream>
#include <range/v3/all.hpp>
#include <string>
using namespace ranges;
int main(int argc, char const *argv[]) {
  auto t = view::ints(0, 10);
  std::cout << "Generated a range from 0 to 10:" << std::endl;
  copy(t, ostream_iterator<>(std::cout, " "));
  std::cout << "\nSliding with step=1 and size=3:" << std::endl;
  auto y = t | slide(1, 3);
  copy(y, ostream_iterator<>(std::cout, "\n"));
  return 0;
}
