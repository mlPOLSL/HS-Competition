#ifndef HEARTSTONE_DATA_MANIPULATOR_HPP
#define HEARTSTONE_DATA_MANIPULATOR_HPP

#include <vector>
#include <json.hpp>
#include <fstream>
#include <Eigen>

/**
 * \brief Class helping heartstone data manipulation for example looking for all unique cards in the data set etc.
 */

class HeartstoneDataManipulator
{
    using json = nlohmann::json;
public:
    enum class CardsType
    {
        All,
        Minion,
        Spell,
        Weapon
    };
    /**
     * \brief Checks for unique cards in the data set and stores them both internally and in output file.
     * \param out Name of the output file where the all unique cards found in the data will be stored. Each card will be stored as json each in new line.
     * \param filenames Vector of paths to the files that make our data set.
     * \param player_hand_only If true considers cards that are in player hand only since played cards for player and opponent differ in parameters. If false takes played_cards as well.
     */
    void generate_unique_cards(const std::string& out, std::vector<std::string> filenames, bool player_hand_only = true);
    /**
     * \brief Calculates the coefficients for calculating true cost of a card for least squares using singular value decomposition
     * \param parameters_names Vector of parameters names that should be considered in evaluation
     * \param filename Name of the file that holds json representations of the cards, each in seperate line
     * \param type What type of cards to consider (spell/weapon/minion/all)
     * \return Vector of double each representing coefficient corresponding to the parameter given (same order)
     */
    std::vector<double> get_coefficients_for_equation(const std::vector<std::string>& parameters_names, const std::string& filename, CardsType type);
    /**
    * \brief Calculates the coefficients for calculating true cost of a card for least squares using singular value decomposition. Differs in using cards
    * that are stored already if genereate_unique_cards() was used, so there is no additional loading from file. Use only if genereate_unique_cards() was used before.
    * \param parameters_names Vector of parameters names that should be considered in evaluation
    * \param type What type of cards to consider (spell/weapon/minion/all)
    * \return Vector of double each representing coefficient corresponding to the parameter given (same order)
    */
    std::vector<double> get_coefficients_for_equation(const std::vector<std::string>& parameters_names, CardsType type);
    double calculate_hand_profitability(const json& hand) const;
    double calculate_card_profitability(const json& card) const;
private:
    inline std::function<bool(const std::string&)> comparator(CardsType type) const;
    void insert_unique_cards_from(const std::vector<json>& j);
    std::vector<json> unique_cards;
    std::vector<double> card_true_value_equation_coefficients;
    std::vector<std::string> parameters_names_acquired;

};



#endif
