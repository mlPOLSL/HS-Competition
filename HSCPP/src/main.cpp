#include <iostream>
#include <json.hpp>
#include <fstream>
#include "../include/HeartstoneDataManipulator.hpp"
using nlohmann::json;

std::vector<std::string> parameters_to_consider{
    "attack","charge","forgetful","freezing","hp","poisonous","shield","stealth","taunt","windfury"
};
std::vector<std::string> parameters_with_intrinsic{
    "attack","charge","forgetful","freezing","hp","poisonous","shield","stealth","taunt","windfury","intrinsic"
};

int main()
{
    HeartstoneDataManipulator hs;
    hs.generate_unique_cards("out100k.txt", { "training/trainingData_JSON_100k.txt" });
    auto i = 0;
    for (auto coefficient : hs.get_coefficients_for_equation(parameters_to_consider,"out100k.txt", HeartstoneDataManipulator::CardsType::Minion))
    {
        std::cout << '\n' << parameters_with_intrinsic[i] << ": " << coefficient;
        ++i;
    } //prints parameter names with corresponding coefficients
    
    std::ifstream input{ "training/trainingData_JSON_100k.txt" };
    std::string temp;
    std::vector<double> profitabilities;
    while(std::getline(input,temp))
        profitabilities.push_back(hs.calculate_hand_profitability(json::parse(temp)["player"]["hand"]));
    std::cout << "---------------------------------------\n";
    std::cout << "\nMax element = " << *std::max_element(profitabilities.cbegin(), profitabilities.cend());
    std::cout << "\nMax element = " << *std::min_element(profitabilities.cbegin(), profitabilities.cend());

    return 0;
}