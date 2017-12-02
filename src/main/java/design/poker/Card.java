package design.poker;

/**
 * Created by tao on 10/17/17.
 */
public class Card {
    private Suit suit;
    private Face face;

    public Card(Suit suit, Face face){
        this.suit=suit;
        this.face=face;
    }

    public Suit getSuit(){
        return suit;
    }

    public Face getFace(){
        return face;
    }

    public int getValue(){
        return face.ordinal()+1;
    }

    public boolean equals(Object o){
        return (o!=null && o instanceof Card && ((Card)o).face==face && ((Card)o).suit==suit);
    }
}
