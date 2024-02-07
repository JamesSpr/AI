import random
import math
import time

class Card:
    # Constructor
    def __init__(self, value, suit):
        self.suit = suit
        self.value = value
    
    # Display   
    def __repr__(self):
        return f"{self.value} of {self.suit}"
    
    def int_value(self, index):
        units = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]
        return units[index]

class Deck:
    def __init__(self, num_decks):
        suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
        self.values = ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Jack", "Queen", "King"]

        self.cards = [Card(value, suit) for value in self.values for suit in suits for _ in range(num_decks)]

    def __repr__(self):
        return f"Deck containing {len(self.cards)} cards"
    
    def __len__(self):
        return len(self.cards)

    def shuffle(self, iterations=1):
        # Shuffle a number of times
        for _ in range(iterations):
            random.shuffle(self.cards)

        # Cut the deck after the shuffle
        split = math.floor(len(self.cards)/2)
        self.cards = self.cards[split:] + self.cards[:split]
    
    def draw(self):
        card = self.cards.pop(0)
        return card
    
    def show(self, num_cards):
        return self.cards[:num_cards]

class Player:
    def __init__(self, id):
        self.id = id
        self.hand = []
        self.actions = []

class Blackjack:
    def __init__(self, rounds, players=1, decks=1):
        self.deck = Deck(decks)
        self.rounds = rounds
        self.dealer = Player("dealer")
        self.players = [Player(player) for player in range(players)]

        # Shuffle the deck before starting
        self.deck.shuffle(5)

    def deal(self):
        for _ in range(2):
            for player in self.players:
                player.hand.append(self.deck.draw())
            
            self.dealer.hand.append(self.deck.draw())

    def reset(self):
        while len(self.dealer.hand) > 0:
            self.deck.cards.append(self.dealer.hand.pop())

        for player in self.players:
            while len(player.hand) > 0:
                self.deck.cards.append(player.hand.pop())

        self.deck.shuffle(3)

    def get_action(self):
        print(f"  What would you like to do (H/S): ", end="")
        action = input().lower()

        while action != "h" and action != "s":
            print(f"  Please enter a valid action: ", end="")
            action = input()

        return action

    def perform_action(self, player, action):
        match action:
            case "h":
                new_card = self.deck.draw()
                player.hand.append(new_card)
                new_value = self.calculate_player_hand(player)
                print(f"  HIT: {new_card} - {new_value}")
                if player.id == "dealer":
                    return

                # Give option for another turn if not over 21
                if new_value > 21:
                    print("  BUST")
                    time.sleep(1)
                    return 
                
                action = self.get_action()
                player.actions.append(action)
                self.perform_action(player, action)
                
            case "s":
                print("  STAND")
            case _:
                print("  INVALID")

        time.sleep(1)


    def play_dealer(self):
        dealer_value = self.calculate_player_hand(self.dealer)
        print(f" Dealers Turn")
        time.sleep(1)
        print(f" Dealers Hole Card: {self.dealer.hand[0]} - {dealer_value}")
        time.sleep(1)

        while dealer_value < 17:
            self.perform_action(self.dealer, "h")

            dealer_value = self.calculate_player_hand(self.dealer)
            time.sleep(1)

        print(f" Dealers Hand: {dealer_value}")
        time.sleep(1)
        return dealer_value

    def calculate_player_hand(self, player):
        hand_value = 0
    
        # Dealer Rule: Dealer must count their first ace as 11
        if player.id == "dealer":
            ace_rule = False
            for card in player.hand:
                card_value = card.int_value(self.deck.values.index(card.value)) 
                if card_value == 1 and not ace_rule:
                    hand_value += 11
                    ace_rule = True
                else:
                    hand_value += card_value
            return hand_value
        
        num_aces = 0
        for card in player.hand:
            card_value = card.int_value(self.deck.values.index(card.value)) 
            if card_value == 1:
                hand_value += 11
                num_aces += 1
            else: 
                hand_value += card_value

        # Count aces as 11, unless that makes the value exceed 21
        if hand_value > 21 and num_aces > 0:
            while num_aces > 0 and hand_value > 21:
                hand_value -= 10
                num_aces -= 1

        return hand_value
    
    def determine_outcome(self, dealer_value):
        print(f"Results:")
        time.sleep(1)
        dealer_value = self.calculate_player_hand(self.dealer)

        for player in self.players:
            player_value = self.calculate_player_hand(player)
            print(f" Player{player.id}: {player_value}", end=" - ")

            if player_value > 21: # Player Bust
                print("LOSS")
            elif dealer_value > 21: # Dealer Bust
                print("WIN")
            elif player_value > dealer_value: # Player Higher
                print("WIN")
            elif player_value < dealer_value: # Dealer Higher
                print("LOSS")
            elif player_value == dealer_value: # Same
                print("DRAW")
            else:
                raise Exception("Unknown Game Outcome")
            
            time.sleep(0.5)

    # Console Application Play
    def play(self):
        for round in range(self.rounds):
            
            print(f"Round {round + 1}")
            self.deal()
            print(f" Dealer shows: {self.dealer.hand[1]}")
            time.sleep(1)

            for player in self.players:
                print(f" It's player {player.id}'s turn:")                
                time.sleep(1)
                print(f"  Player {player.id} hand: {self.calculate_player_hand(player)} - {player.hand}")
                time.sleep(1)
                action = self.get_action()
                player.actions.append(action)
                self.perform_action(player, action)

            time.sleep(1)
            dealer_value = self.play_dealer()
            self.determine_outcome(dealer_value)
            print()

            time.sleep(1)
            self.reset()

if __name__ == "__main__":
    blackjack = Blackjack(rounds=5, players=2, decks=1)
    blackjack.play()