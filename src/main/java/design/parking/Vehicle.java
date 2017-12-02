package design.parking;

import java.util.ArrayList;

/**
 * Created by tao on 10/24/17.
 */
public abstract class Vehicle {
    protected  ArrayList<ParkingSpot> parkingSpots = new ArrayList<>();
    protected  String licensePlate;
    protected  int spotsNeeded;
    protected VehicleSize size;
    public int getSpotsNeeded(){
        return spotsNeeded;
    }

    public VehicleSize getSize(){
        return size;
    }

    public void parkInSpot(ParkingSpot s){
        parkingSpots.add(s);
    }

    //remove car from spot
    public void clearSpots(){
        for(int i=0;i<parkingSpots.size();++i)
            parkingSpots.get(i).removeVehicle();
        parkingSpots.clear();
    }

    public abstract  boolean canFitInSpot(ParkingSpot spot);

    public abstract void print();
}


