package design.poker;

import design.poker.Card;
import design.poker.Face;
import design.poker.Suit;

import java.util.ArrayList;
import java.util.Random;

/**
 * Created by tao on 10/17/17.
 */
public class CardsDeck {

    private ArrayList<Card>mCards;
    private ArrayList<Card>mPulledCards;
    private Random mRandom;
    public CardsDeck(){
        mRandom = new Random();
        mPulledCards = new ArrayList<>();
        mCards = new ArrayList<>(Suit.values().length* Face.values().length);
        reset();
    }

    public void reset(){
        mCards.clear();
        mPulledCards.clear();
        for(Suit s: Suit.values()){
            for( Face f:Face.values()){
                Card c = new Card(s,f);
                mCards.add(c);
            }
        }
    }

    public Card pullRandom(){
        if(mCards.isEmpty())
            return null;
        Card res = mCards.remove(mRandom.nextInt(mCards.size()));
        if(res!=null)
            mPulledCards.add(res);
        return res;
    }

    public Card getRandom(){
        if(mCards.isEmpty())
            return null;
        Card  c = mCards.get(mRandom.nextInt(mCards.size()));
        return c;
    }

    public boolean isEmpty(){
        return mCards.isEmpty();
    }

}
