import com.espertech.esper.common.client.EPCompiled;
import com.espertech.esper.common.client.EventBean;
import com.espertech.esper.common.client.configuration.Configuration;
import com.espertech.esper.common.internal.event.map.MapEventBean;
import com.espertech.esper.compiler.client.CompilerArguments;
import com.espertech.esper.compiler.client.EPCompileException;
import com.espertech.esper.compiler.client.EPCompiler;
import com.espertech.esper.compiler.client.EPCompilerProvider;
import com.espertech.esper.runtime.client.*;
import org.antlr.v4.runtime.misc.Pair;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVPrinter;
import org.apache.commons.csv.CSVRecord;
import org.apache.log4j.BasicConfigurator;

import java.io.*;
import java.util.*;


public class Main {
    final static String scoresPath = "Data/scores.txt";

    final static String trainStreamPath = "Data/train data stream.txt";
    final static String trainMatchesPath = "Application/output data/matches/train matches";
    final static String trainElaboratedMatchesPath = "Application/output data/matches/train elaborated matches";
    final static int trainEventsNum = 6077571;

    final static String testStreamPath = "Data/test data stream.txt";
    final static int testEventsNum = 1000000;

    final static String testMatchesPathPrefix = "Application/output data/matches/test matches";
    final static String testElaboratedMatchesPathPrefix = "Application/output data/matches/test elaborated matches";

    final static String testDetailsPathPrefix = "Application/output data/details/test details";

    final static String trainDetailsPathPrefix = "Application/output data/details/train details";

    final static String testThresholdDetailsPathPrefix = "Application/output data/details/threshold details/details";

    final static String testMaximumKDetailsPathPrefix = "Application/output data/details/maximumk details/details";

    final static String patternPath = "Application/pattern";

    final static Map< Integer, List<List<Integer>> > idToMatches = new HashMap<>();
    final static Set< Set<Integer> > matches = new HashSet<>();

    final static String statementName = "mystatement";
    final static int toPrint = 1000;
    final static boolean printProgress = true;
    final static int windowSize = 8;
    static int matchesNum = 0;

    static boolean printTime = true;
    static int toPrintTime = 10000;

    static boolean countMultipleMatchesOfSameEvents = true;

    static int startIndex = 0;
    static int endIndex = testEventsNum;

    static EPRuntime runtime;

    static int currentMatches;

    public static UpdateListener listener = (newData, oldData, s, r) -> {
        for (EventBean match : newData) {
            Map<String, Integer> events = (Map<String, Integer>)(Map<String, ?>)((MapEventBean) match).getProperties();
            List<Integer> matchCounts = new LinkedList<>(events.values());
            if(countMultipleMatchesOfSameEvents) {
                processMatch(matchCounts);
            } else {
                final Set<Integer> matchCountsSet = new HashSet<>(matchCounts);
                if (!matches.contains(matchCountsSet)) {
                    matches.add(matchCountsSet);
                    processMatch(matchCounts);
                }
            }
        }
    };

    private static void processMatch(List<Integer> matchCounts) {
        matchesNum += 1;

        for (Integer c : matchCounts) {
            List<List<Integer>> l = idToMatches.get(c);
            l.add(matchCounts);
        }
    }

    public static UpdateListener listenerForBestSubsets = (newData, oldData, s, r) -> currentMatches += newData.length;


    public static void main(String[] s) {
        BasicConfigurator.configure();
        try {
            final var outputPath = "Application/output_data/best subsets train.txt";
            optimalSubsets(trainStreamPath, outputPath);
        } catch (IOException ex){
            ex.printStackTrace();
        }
    }

    private static void queryTrainData() throws IOException {
        query(trainStreamPath, trainMatchesPath, trainElaboratedMatchesPath, trainDetailsPathPrefix,
                null, trainEventsNum, null, null, true);
    }

    private static void queryTestData() throws IOException {
        query(testStreamPath, testMatchesPathPrefix, testElaboratedMatchesPathPrefix, testDetailsPathPrefix,
                null, testEventsNum, null, null, true);
    }

    private static void testThresholds() throws IOException {
        double threshold = 0.0;
        for (int i = 0; i < 10; i++) {
            query(testStreamPath, testMatchesPathPrefix, testElaboratedMatchesPathPrefix, testThresholdDetailsPathPrefix,
                    scoresPath, testEventsNum, threshold, null, false);
            threshold += 0.12;
        }
    }

    private static void testMaximumK() throws IOException {
        final var threshold = 0.12;
        int[] ks = new int[]{5, 6, 7, 8};
        for (int k : ks) {
            query(testStreamPath, testMatchesPathPrefix, testElaboratedMatchesPathPrefix, testMaximumKDetailsPathPrefix,
                    scoresPath, testEventsNum, threshold, k, true);
        }
    }

    private static void query(String inputPath, String outputPath, String elaboratedOutputPath, String detailsPath,
                              String scoresPath, int eventsNum, Double threshold, Integer k, boolean printMatches)
            throws IOException {

        for (int count = startIndex; count < endIndex; count++) {
            idToMatches.put(count, new LinkedList<>());
        }

        boolean printThreshold = true;
        if(threshold == null){
            threshold = (double) 0;
            printThreshold = false;
        }

        runtime = getEpRuntime(listener);

        int sentEventsNum = 0;
        if(k != null){
            sentEventsNum = sendEventsMaximumK(inputPath, scoresPath, threshold, k);
        }
        else {
            sentEventsNum = sendEvents(inputPath, scoresPath, threshold);
        }


        String kString = "_";
        String label = Double.toString(threshold);
        if(k != null){
            kString = "_k_" + (double) k;
            label = Integer.toString(k);
        }

        var s = "";
        if(printThreshold) {
            s = "_threshold_" + threshold;
        }

        if(k == null && !printThreshold){
            kString = "";
        }

        String postfix  = s + kString + ".txt";
        CSVPrinter detailsPrinter;
        if(startIndex != 0){
            detailsPrinter = new CSVPrinter(new FileWriter(detailsPath + postfix, true), CSVFormat.DEFAULT);
        } else {
            detailsPrinter = new CSVPrinter(new FileWriter(detailsPath + postfix), CSVFormat.DEFAULT);
        }
        detailsPrinter.printRecord(label);
        detailsPrinter.printRecord(sentEventsNum);
        detailsPrinter.printRecord(matchesNum);
        detailsPrinter.close();

        System.out.println();
        System.out.println(label);
        System.out.println(sentEventsNum);
        System.out.println(matchesNum);

        if(printMatches) {
            outputPath = outputPath + postfix;
            elaboratedOutputPath = elaboratedOutputPath + postfix;
            CSVPrinter writer;
            CSVPrinter elaboratedWriter;
            if(startIndex != 0){
                writer = new CSVPrinter(new FileWriter(outputPath, true), CSVFormat.DEFAULT);
                elaboratedWriter = new CSVPrinter(new FileWriter(elaboratedOutputPath, true), CSVFormat.DEFAULT);
            } else {
                writer = new CSVPrinter(new FileWriter(outputPath), CSVFormat.DEFAULT);
                elaboratedWriter = new CSVPrinter(new FileWriter(elaboratedOutputPath), CSVFormat.DEFAULT);
            }


            for (int count = startIndex; count < endIndex; count++) {
                List<List<Integer>> matches = idToMatches.get(count);
                writer.print(matches.size() + ", " + count);
                writer.println();
                elaboratedWriter.print(matches.size() + " ; ");

                for (List<Integer> match : matches) {
                    elaboratedWriter.print(match + " ; ");
                }
                elaboratedWriter.println();
            }
            writer.close();
            elaboratedWriter.close();
        }
        idToMatches.clear();
        matches.clear();
        matchesNum = 0;
        runtime.destroy();
        System.out.println("Done");
    }

    private static EPRuntime getEpRuntime(UpdateListener l) {
        EPCompiler compiler = EPCompilerProvider.getCompiler();
        Configuration configuration = new Configuration();
        configuration.getCommon().addEventType(Event.class);
        CompilerArguments args = new CompilerArguments(configuration);

        EPCompiled epCompiled;
        String pattern;
        try {
            pattern = getPattern();
            epCompiled = compiler.compile("@name('" + statementName + "') " + pattern, args);
        }
        catch (EPCompileException ex) {
            throw new RuntimeException(ex);
        }

        EPRuntime runtime = EPRuntimeProvider.getDefaultRuntime(configuration);
        EPDeployment deployment;
        try {
            deployment = runtime.getDeploymentService().deploy(epCompiled);
        }
        catch (EPDeployException ex) {
            throw new RuntimeException(ex);
        }

        EPStatement statement =
                runtime.getDeploymentService().getStatement(deployment.getDeploymentId(), statementName);

        statement.addListener(l);
        return runtime;
    }

    private static String getPattern() {
        StringBuilder pattern = new StringBuilder();
        BufferedReader reader;
        try {
            reader = new BufferedReader(new FileReader(patternPath));
            String firstLine = reader.readLine();
            pattern.append(firstLine);
            String line = reader.readLine();
            while (line != null && !line.contentEquals("done")) {
                pattern.append(" ").append(line);
                line = reader.readLine();
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return pattern.toString();
    }

    static boolean eventFilter(float score, double threshold){
        return score >= threshold || score == -1;
    }

    static int sendEvents(String inputPath, String scoresPath, double threshold) throws IOException {
        CSVParser eventsParser = CSVParser.parse(new FileReader(inputPath), CSVFormat.DEFAULT);

        CSVParser scoresParser = null;
        Iterator<CSVRecord> scores = null;

        int notSent = 0;
        Event dummyEvent = new Event("-1", -1, -1, -1);

        String score = null;

        if(scoresPath != null){
            scoresParser = CSVParser.parse(new FileReader(scoresPath), CSVFormat.DEFAULT);
            scores = scoresParser.iterator();
            score = scores.next().get(0);
        }

        long startTime = System.currentTimeMillis();

        Iterator<CSVRecord> events = eventsParser.iterator();

        for (int i = 0; i < startIndex; i++) {
            events.next();
        }

        int count = startIndex;

        while(events.hasNext()) {
            CSVRecord s = events.next();

            Event eventForUse;
            if(score == null || eventFilter(Float.parseFloat(score), threshold)) {
                eventForUse = new Event(s.get(0), Double.parseDouble(s.get(1)), count, count);
            }
            else {
                notSent += 1;
                eventForUse = dummyEvent;
            }
            runtime.getEventService().sendEventBean(eventForUse, "Event");

            count += 1;

            if(printProgress && (count % toPrint == 0)){
                System.out.println(count);
            }

            if(printTime && (count % toPrintTime == 0)){
                long currentTime = System.currentTimeMillis();
                double secPassed = (currentTime - startTime)/1000;
                double minPassed = secPassed/60;
                System.out.println("Time passed: " + secPassed + " secs");
                System.out.println("Time passed: " + minPassed + " mins");
                System.out.println("Current matches num: " + matchesNum);
            }

            if(count == endIndex){
                break;
            }

            if(scoresParser != null) {
                score = scores.next().get(0);
            }
        }

        eventsParser.close();
        if(scoresParser != null){
            scoresParser.close();
        }
        return testEventsNum - notSent;
    }

    static int sendEventsMaximumK(String inputPath, String scoresPath, double threshold, int k)
            throws IOException{

        CSVParser eventsParser = CSVParser.parse(new FileReader(inputPath), CSVFormat.DEFAULT);
        Iterator<CSVRecord> events = eventsParser.iterator();

        CSVParser scoresParser = CSVParser.parse(new FileReader(scoresPath), CSVFormat.DEFAULT);
        Iterator<CSVRecord> scores = scoresParser.iterator();

        Event dummyEvent = new Event("-1", -1, -1, -1);

        int count = 0;
        int id = 0;

        List<Pair<Event, Float>> window = new LinkedList<>();
        for (int i = 0; i < windowSize; i++) {
            CSVRecord e = events.next();
            Event event = new Event(e.get(0), Double.parseDouble(e.get(1)), count, id);

            String score = scores.next().get(0);
            window.add(new Pair<>(event, Float.parseFloat(score)));

            count += 1;
            id += 1;
        }

        while (events.hasNext()) {
            sendBestK(threshold, k, dummyEvent, window);
            count = createPartition(dummyEvent, count, window);

            window.remove(0);

            count += 1;

            CSVRecord e = events.next();
            String score = scores.next().get(0);

            Event event = new Event(e.get(0), Double.parseDouble(e.get(1)), count, id);
            window.add(new Pair<>(event, Float.parseFloat(score)));

            id += 1;
            if(printProgress && (id % toPrint == 0)){
                System.out.println(id);
            }
        }

        scoresParser.close();
        eventsParser.close();
        return testEventsNum;
    }

    private static void sendBestK(double threshold, Integer k, Event dummyEvent, List<Pair<Event, Float>> window) {
        window.sort(Comparator.comparing(p -> p.b)); //sort by score
        Collections.reverse(window);

        var iterator = window.listIterator();
        List<Pair<Event, Float>> topK = new LinkedList<>();
        for (int j = 0; j < k; j++) {
            final var next = iterator.next();
            topK.add(next);
        }

        topK.sort(Comparator.comparing(p -> p.a.getCount()));
        var iterator2 = topK.listIterator();

        for (int j = 0; j < k; j++) {
            final var next = iterator2.next();
            Event event = next.a;
            Float score = next.b;
            Event eventForUse;
            if (eventFilter(score, threshold)) {
                eventForUse = event;
            } else {
                eventForUse = dummyEvent;
            }
            runtime.getEventService().sendEventBean(eventForUse, "Event");
        }

        window.sort(Comparator.comparing(p -> p.a.getCount()));
    }

    public static <T> ArrayList<List<T>> getSubsets(List<T> s) {
        ArrayList<T> set = new ArrayList<>(s);
        ArrayList<List<T>> allsubsets =
                new ArrayList<List<T>>();
        int max = 1 << set.size();
        for (int i = 0; i < max; i++) {
            ArrayList<T> subset = new ArrayList<T>();
            for (int j = 0; j < set.size(); j++) {
                if (((i >> j) & 1) == 1) {
                    subset.add(set.get(j));
                }
            }
            allsubsets.add(new LinkedList<>(subset));
        }
        return allsubsets;
    }

    static void optimalSubsets(String inputPath, String outputPath)
            throws IOException{
        CSVParser eventsParser = CSVParser.parse(new FileReader(inputPath), CSVFormat.DEFAULT);
        Iterator<CSVRecord> events = eventsParser.iterator();

        CSVPrinter printer = new CSVPrinter(new FileWriter(outputPath, false), CSVFormat.DEFAULT);

        runtime = getEpRuntime(listenerForBestSubsets);

        Event dummyEvent = new Event("-1", -1, -1, -1);

        int count = 0;
        int id = 0;

        long startTime = System.currentTimeMillis();
        long lastTime = startTime;

        List<Event> window = new LinkedList<>();
        for (int i = 0; i < windowSize; i++) {
            CSVRecord e = events.next();
            Event event = new Event(e.get(0), Double.parseDouble(e.get(1)), count, id);

            window.add(event);

            count += 1;
            id += 1;
        }

        while (events.hasNext()) {
            ArrayList< List<Event> > subsets = getSubsets(window);
            ArrayList<Integer> bestSubsetsMatches = new ArrayList<>(Collections.nCopies(windowSize, 0));

            for (var subset : subsets) {
                int k = subset.size() - 1;
                if(k == -1){
                    continue;
                }

                currentMatches = 0;

                for (Event e : subset) {
                    runtime.getEventService().sendEventBean(e, "Event");
                }

                for (int i = 0; i < windowSize; i++) {
                    runtime.getEventService().sendEventBean(dummyEvent, "Event");
                }

                if(currentMatches > bestSubsetsMatches.get(k)){
                    bestSubsetsMatches.set(k, currentMatches);
                }
            }

            printer.printRecord(bestSubsetsMatches);

            window.remove(0);

            CSVRecord e = events.next();

            Event event = new Event(e.get(0), Double.parseDouble(e.get(1)), count, id);
            window.add(event);

            count += 1;
            id += 1;
            if(printProgress && (id % toPrint == 0)){
                System.out.println(id);
            }

            if(printTime && (count % toPrintTime == 0)){
                long currentTime = System.currentTimeMillis();
                long secPassed = (currentTime - startTime)/1000;
                double minPassed = secPassed/60;
                long thisWindowTime = (currentTime - lastTime)/1000;
                System.out.println("Time passed: " + secPassed + " secs");
                System.out.println("Time passed: " + minPassed + " mins");
                System.out.println("This window's time: " + thisWindowTime);
                lastTime = currentTime;
            }
        }

        printer.close();
        eventsParser.close();
    }

    private static int createPartition(Event dummyEvent, int count, List<Pair<Event, Float>> window) {
        for (int i = 0; i < windowSize; i++) {
            runtime.getEventService().sendEventBean(dummyEvent, "Event");
        }

        int jumpSize = 2*windowSize - 1;
        count += jumpSize;

        for (var p : window) {
            Event ev = p.a;
            ev.setCount(ev.getCount() + jumpSize);
        }
        return count;
    }
}
