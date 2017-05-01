#include "../include/HeartstoneDataManipulator.hpp"

void HeartstoneDataManipulator::generate_unique_cards(const std::string& out, std::vector<std::string> filenames, bool player_hand_only)
{
    std::ifstream input;
    std::ofstream output{ out };
    std::string temp;
    for (auto& filename : filenames)
    {
        
        input.open(filename);
        while (std::getline(input, temp)) 
        {
            json j = json::parse(temp);
            if (player_hand_only)
            {
                insert_unique_cards_from(j["player"]["hand"]);
            }
            else
            {
                insert_unique_cards_from(j["opponent"]["played_cards"]);
                insert_unique_cards_from(j["player"]["hand"]);
                insert_unique_cards_from(j["player"]["played_cards"]);
            }
        }
        input.close();
    }
    for (auto& card : unique_cards)
        output << card.dump(-1) << '\n';
    output.close();
}


std::vector<double> HeartstoneDataManipulator::get_coefficients_for_equation(const std::vector<std::string>& parameters_names, const std::string& filename, CardsType type)
{
    parameters_names_acquired = parameters_names;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cards_parameters_matrix;
    auto parameters_count = parameters_names.size();
    cards_parameters_matrix.resize(0, parameters_count+1); // +1 because of last intrinsic value
    Eigen::VectorXd cards_mana_cost_vector;
    std::string temp;

    std::ifstream input{ filename };
    while (std::getline(input, temp))
    {
        auto card = json::parse(temp);
        if (comparator(type)(card["type"]))
        {
            cards_parameters_matrix.conservativeResize(cards_parameters_matrix.rows() + 1, Eigen::NoChange);
            cards_mana_cost_vector.conservativeResize(cards_mana_cost_vector.rows() + 1, Eigen::NoChange);
            Eigen::RowVectorXd temp_vector;
            temp_vector.resize(1, parameters_count + 1);
            for (auto i = 0; i < parameters_count; ++i)
            {
                if (card[parameters_names[i]].is_boolean())
                {
                    temp_vector(0, i) = static_cast<double>(card[parameters_names[i]].get<bool>());
                }
                else
                {
                    temp_vector(0, i) = card[parameters_names[i]].get<double>();
                }
            }

            temp_vector(0, parameters_count) = 1; //adding last parameter which is intrinsic value
            cards_parameters_matrix.row(cards_parameters_matrix.rows() - 1) = temp_vector;
            cards_mana_cost_vector(cards_mana_cost_vector.rows() - 1) = card["crystals_cost"].get<double>();
        }        
    }
    input.close();
    Eigen::VectorXd coefficients = cards_parameters_matrix.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(cards_mana_cost_vector);
    std::vector<double> obtained_coefficients(coefficients.rows());
    for (int i = 0; i < coefficients.rows(); ++i)
       obtained_coefficients[i] = coefficients(i, 0);
    card_true_value_equation_coefficients = obtained_coefficients;
    return obtained_coefficients;
}



std::vector<double> HeartstoneDataManipulator::get_coefficients_for_equation(const std::vector<std::string>& parameters_names, CardsType type)
{
    parameters_names_acquired = parameters_names;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> cards_parameters_matrix;
    auto parameters_count = parameters_names.size();
    cards_parameters_matrix.resize(0, parameters_count + 1);
    Eigen::VectorXd cards_mana_cost_vector;
    std::string temp;
    for(const auto& card : unique_cards)
    {
        if (comparator(type)(card["type"]))
        {
            cards_parameters_matrix.conservativeResize(cards_parameters_matrix.rows() + 1, Eigen::NoChange);
            cards_mana_cost_vector.conservativeResize(cards_mana_cost_vector.rows() + 1, Eigen::NoChange);
            Eigen::RowVectorXd temp_vector;
            temp_vector.resize(1, parameters_count + 1);
            for (auto i = 0; i < parameters_count; ++i)
            {
                if (card[parameters_names[i]].is_boolean())
                {
                    temp_vector(0, i) = static_cast<double>(card[parameters_names[i]].get<bool>());
                }
                else
                {
                    temp_vector(0, i) = card[parameters_names[i]].get<double>();
                }
            }

            temp_vector(0, parameters_count) = 1; //adding last parameter which is intrinsic value
            cards_parameters_matrix.row(cards_parameters_matrix.rows() - 1) = temp_vector;
            cards_mana_cost_vector(cards_mana_cost_vector.rows() - 1) = card["crystals_cost"].get<double>();
        }
    }
    Eigen::VectorXd coefficients = cards_parameters_matrix.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(cards_mana_cost_vector);
    std::vector<double> obtained_coefficients(coefficients.rows());
    for (int i = 0; i < coefficients.rows(); ++i)
        obtained_coefficients[i] = coefficients(i, 0);
    card_true_value_equation_coefficients = obtained_coefficients;
    return obtained_coefficients;
}

double HeartstoneDataManipulator::calculate_hand_profitability(const json& hand) const
{
    double overall_profitability = 0.0;
    for(const auto& card : hand)
        overall_profitability += calculate_card_profitability(card);
    return overall_profitability;
}

double HeartstoneDataManipulator::calculate_card_profitability(const json& card) const
{
    
    if (card["type"] == "MINION")
    {
        auto card_cost = card["crystals_cost"].get<double>();
        auto card_true_cost = 0.0;
        for (auto i = 0u; i < parameters_names_acquired.size(); ++i)
        {
            card_true_cost += card_true_value_equation_coefficients[i] * (card[parameters_names_acquired[i]].is_boolean() ? static_cast<double>(card[parameters_names_acquired[i]].get<bool>()) : card[parameters_names_acquired[i]].get<double>());
        }
        card_true_cost += card_true_value_equation_coefficients.back() * 1;
        return card_cost - card_true_cost;
    }
    else if (card["type"] == "SPELL")
        return 0;
    return 0;
}

std::function<bool(const std::string&)> HeartstoneDataManipulator::comparator(CardsType type) const
{
    if (type == CardsType::Minion)
        return [](const std::string& s) {return (s == "MINION") ? true : false; };
    else if (type == CardsType::Spell)
        return [](const std::string& s) {return (s == "SPELL") ? true : false; };
    else if (type == CardsType::Weapon)
        return [](const std::string& s) {return (s == "WEAPON") ? true : false; };
    else
        return [](const std::string& s) {return true; };
}

void HeartstoneDataManipulator::insert_unique_cards_from(const std::vector<json>& j)
{
    for (auto& card : j)
    {
        if (!std::any_of(unique_cards.cbegin(), unique_cards.cend(), [&card](const json& unique_card)
        {
            if (unique_card["name"] == card["name"])
                return true;
            return false;
        })) unique_cards.emplace_back(card);
    }
}
