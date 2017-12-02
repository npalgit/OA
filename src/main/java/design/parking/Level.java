package design.parking;

/**
 * Created by tao on 10/24/17.
 */
public class Level {
    private int floor;
    private ParkingSpot[] spots;
    private int availableSpots = 0;
    private static final int SPOTS_PER_ROW =10;
    public Level(int flr, int numberSpots){
        this.floor = flr;
        this.availableSpots = numberSpots;
        this.spots = new ParkingSpot[numberSpots];
        int largeSpots = numberSpots/4;
        int bikeSpots = numberSpots/4;
        int compactSpots = numberSpots-largeSpots-bikeSpots;
        for(int i=0;i<numberSpots;++i){
            VehicleSize sz = VehicleSize.Motorcycle;
            if(i<largeSpots)
                sz = VehicleSize.Large;
            else if(i<largeSpots+compactSpots)
                sz = VehicleSize.Compact;
            int row = i/SPOTS_PER_ROW;
            spots[i]= new ParkingSpot(this,row,i,sz);
        }
    }

    public int getAvailableSpots(){
        return availableSpots;
    }


    //find a place to park this vehicle, return false if failed
    public boolean parkVehicle(Vehicle vehicle){
        if(getAvailableSpots()<vehicle.getSpotsNeeded())
            return false;
        int spotNumber = findAvailableSpots(vehicle);
        if(spotNumber<0)
            return false;
        return parkStartingAtSpot(spotNumber,vehicle);
    }


    //park a vehicle starting at the spot spotNumber, and continuing until vehicle.spotsNeeded
    public boolean parkStartingAtSpot(int num, Vehicle v){
        v.clearSpots();
        boolean success = true;
        for(int i=num;i<num+v.spotsNeeded;++i){
            success &= spots[i].park(v);
        }
        availableSpots -=v.spotsNeeded;
        return success;
    }

    //find a spot to park this vehicle, return index of spot, or -1 on failure
    public int findAvailableSpots(Vehicle vehicle){
        int spotsNeeded = vehicle.getSpotsNeeded();
        int lastRow = -1;
        int spotsFound = 0;
        for(int i=0;i<spots.length;++i){
            ParkingSpot  spot = spots[i];
            if(lastRow!=spot.getRow()){
                spotsFound = 0;
                lastRow = spot.getRow();
            }
            if(spot.canFitVehicle(vehicle)){
                spotsFound++;
            }else
                spotsFound=0;
            if(spotsFound==spotsNeeded)
                return i-(spotsNeeded-1);
        }
        return -1;
    }

    //when a car was removed from the spot, increment availablespots;
    public void spotFreed(){
        availableSpots++;
    }
    public void print(){
        int lastRow = -1;
        for(int i=0;i<spots.length;++i){
            ParkingSpot spot = spots[i];
            if(spot.getRow()!=lastRow){
                System.out.print("  ");
                lastRow = spot.getRow();
            }
            spot.print();
        }

    }
}
